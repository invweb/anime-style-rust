use anyhow::Result;
use axum::{
    extract::Multipart,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Router,
};
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use std::sync::Arc;
use futures::stream::Stream;
use tokio::sync::mpsc;

mod image_utils;
mod model;

struct AppState {
    model: Mutex<model::AnimeModel>,
}

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../static/index.html"))
}

async fn upload(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Vec<u8>, (StatusCode, String)> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();
        if name == "image" {
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

            let img = image::load_from_memory(&data)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

            let resized = image_utils::resize(&img, 512);
            let (w, h) = (resized.width() as usize, resized.height() as usize);
            let tensor = image_utils::rgb_to_tensor(&resized)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            let mut model = state.model.lock().await;
            let result_tensor = model
                .run("anime style, high quality, detailed", Some(&tensor), w, h, 0.75, None)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            let result = image_utils::tensor_to_rgb(&result_tensor, w, h);

            let mut buf = std::io::Cursor::new(Vec::new());
            result
                .write_to(&mut buf, image::ImageFormat::Png)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            return Ok(buf.into_inner());
        }
    }

    Err((StatusCode::BAD_REQUEST, "No image field found".into()))
}

async fn upload_sse(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::spawn(async move {
        while let Some(field) = multipart.next_field().await.transpose() {
            let field = match field {
                Ok(f) => f,
                Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; return; }
            };
            let name = field.name().unwrap_or_default().to_string();
            if name != "image" { continue; }

            let data = match field.bytes().await {
                Ok(d) => d,
                Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; return; }
            };

            let img = match image::load_from_memory(&data) {
                Ok(i) => i,
                Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; return; }
            };

            let resized = image_utils::resize(&img, 512);
            let (w, h) = (resized.width() as usize, resized.height() as usize);
            let tensor = match image_utils::rgb_to_tensor(&resized) {
                Ok(t) => t,
                Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; return; }
            };

            let _ = tx.send(Ok(Event::default().event("progress").data("encoding"))).await;

            let mut model = state.model.lock().await;
            let tx_clone = tx.clone();
            let result_tensor = model.run(
                "anime style, high quality, detailed",
                Some(&tensor),
                w, h, 0.75,
                Some(&move |step: usize, total: usize, dt: f32| {
                    let msg = format!("{}|{}|{:.1}", step, total, dt);
                    let _ = tx_clone.try_send(Ok(Event::default().event("step").data(msg)));
                }),
            );

            let result_tensor = match result_tensor {
                Ok(t) => t,
                Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; return; }
            };

            let _ = tx.send(Ok(Event::default().event("progress").data("decoding"))).await;

            let result = image_utils::tensor_to_rgb(&result_tensor, w, h);
            let mut buf = std::io::Cursor::new(Vec::new());
            if let Err(e) = result.write_to(&mut buf, image::ImageFormat::Png) {
                let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await;
                return;
            }

            let png_data = buf.into_inner();
            let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png_data);
            let _ = tx.send(Ok(Event::default().event("done").data(b64))).await;
        }
    });

    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
}

async fn health() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "./models/stable-diffusion-v1-5".into());
    let steps: usize = std::env::var("STEPS")
        .unwrap_or_else(|_| "20".into())
        .parse()
        .unwrap_or(20);
    let use_f16 = std::env::var("F16")
        .unwrap_or_else(|_| "1".into())
        == "1";

    println!("Loading model from: {model_dir} (f16={use_f16})");
    let anime_model = model::AnimeModel::new(&model_dir, steps, use_f16, 512, 512)?;

    let state = Arc::new(AppState {
        model: Mutex::new(anime_model),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/process", post(upload))
        .route("/process-stream", post(upload_sse))
        .route("/health", get(health))
        .with_state(state);

    let addr: SocketAddr = "[::]:3000".parse().unwrap();
    println!("Server running at http://{addr}");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
