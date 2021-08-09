// reference: https://github.com/tensorflow/rust/blob/master/examples/addition.rs
use std::result::Result;
use std::error::Error;

mod embeddings;
use embeddings::Embeddings;

pub fn main() -> Result<(), Box<dyn Error>> {
    // made by make_model_file.py
    let filename = "model/model.pb";

    // embedding
    let emb = Embeddings::new(&filename);
    let emb_vec = emb.convert_from_img("img/example.jpeg")?;
    println!("{:?}", emb_vec);

    Ok(())
}
