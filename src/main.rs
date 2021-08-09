// reference: https://github.com/tensorflow/rust/blob/master/examples/addition.rs
use std::result::Result;
use std::error::Error;
use std::fs;
use std::collections::HashMap;

use hora::core::ann_index::ANNIndex;

mod embeddings;
use embeddings::Embeddings;

pub fn main() -> Result<(), Box<dyn Error>> {
    // made by make_model_file.py
    let filename = "model/model.pb";

    // init embedding model
    let emb = Embeddings::new(&filename);

    // init index
    let mut index = hora::index::hnsw_idx::HNSWIndex::<f32, usize>::new(
        2048,
        &hora::index::hnsw_params::HNSWParams::<f32>::default(),
    );

    // add point
    let paths = fs::read_dir("img")?;
    let mut file_map = HashMap::new();
    for (i, path) in paths.into_iter().enumerate() {
        let file_path = path?.path();
        let path_str = file_path.to_str();
        if path_str.is_some() {
            file_map.insert(i, path_str.unwrap().to_string().clone());  // key: id, value: filename
            let emb_vec = emb.convert_from_img(path_str.unwrap())?;     // convert embedding vector
            index.add(emb_vec.as_slice(), i)?;                          // indexing
            println!("index file {:?}: {:?}", i, path_str.unwrap());
        }
    }
    index.build(hora::core::metrics::Metric::CosineSimilarity).unwrap();

    // search
    let query_image = file_map[100]                            // select search query
    let emb_vec_target = emb.convert_from_img(query_image)?;   // convert embedding vector
    let result = index.search(emb_vec_target.as_slice(), 10);  // search
    println!("neighbor images by query: {:?}", query_image);
    for r in result {
        println!("{:?}", file_map[r]);
    }

    Ok(())
}
