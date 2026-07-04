use anyhow::Result;
use axum::{
    extract::{Multipart, Json},
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Router,
};
use serde::Deserialize;
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

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
}

type SharedState = Arc<AppState>;

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../static/index.html"))
}

fn sse_generate(tx: mpsc::Sender<Result<Event, Infallible>>, prompt: String, state: SharedState) {
    tokio::spawn(async move {
        let _ = tx.send(Ok(Event::default().event("progress").data("encoding"))).await;

        let mut model = state.model.lock().await;
        let tx_clone = tx.clone();
        let result_tensor = model.run(
            &prompt,
            None,
            512, 512, 0.75,
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

        let result = image_utils::tensor_to_rgb(&result_tensor, 512, 512);
        let mut buf = std::io::Cursor::new(Vec::new());
        if let Err(e) = result.write_to(&mut buf, image::ImageFormat::Png) {
            let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await;
            return;
        }

        let png_data = buf.into_inner();
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png_data);
        let _ = tx.send(Ok(Event::default().event("done").data(b64))).await;
    });
}

fn sse_process(tx: mpsc::Sender<Result<Event, Infallible>>, img_bytes: Vec<u8>, state: SharedState) {
    tokio::spawn(async move {
        let img = match image::load_from_memory(&img_bytes) {
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
    });
}

async fn generate_stream(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    sse_generate(tx, req.prompt, Arc::clone(&state));
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
}

async fn upload_stream(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    while let Some(field) = multipart.next_field().await.transpose() {
        let field = match field {
            Ok(f) => f,
            Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; break; }
        };
        let name = field.name().unwrap_or_default().to_string();
        if name != "image" { continue; }

        let data = match field.bytes().await {
            Ok(d) => d.to_vec(),
            Err(e) => { let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await; break; }
        };

        sse_process(tx.clone(), data, Arc::clone(&state));
        break;
    }

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
        .route("/generate-stream", post(generate_stream))
        .route("/process-stream", post(upload_stream))
        .route("/health", get(health))
        .with_state(state);

    let addr: SocketAddr = "[::]:3000".parse().unwrap();
    println!("Server running at http://{addr}");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
