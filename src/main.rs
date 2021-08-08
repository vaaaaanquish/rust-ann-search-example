// reference: https://github.com/tensorflow/rust/blob/master/examples/addition.rs
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;


pub fn main() -> Result<(), Box<dyn Error>> {
    // made by make_model_file.py
    let filename = "model/model.pb";

    // load model file
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;

    // input vector
    let x = Tensor::new(&[1, 224, 224, 3]).with_values(&[3.0f32; 224*224*3])?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("resnet")?, 0, &x);
    let output = args.request_fetch(&graph.operation_by_name_required("Identity")?, 0);
    session.run(&mut args)?;

    // check result
    let output_tensor: Tensor<f32> = args.fetch(output)?;
    let output_array: Vec<f32> = output_tensor.iter().map(|x| x.clone()).collect();
    println!("{:?}", output_array);

    Ok(())
}
