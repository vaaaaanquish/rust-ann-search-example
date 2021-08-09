use std::result::Result;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;
use image::io::Reader as ImageReader;
use image::imageops::FilterType;

pub struct Embeddings {
    session: Session,
    graph: Graph,
}

impl Embeddings {
    #[allow(unused_must_use)]
    pub fn new(model_path: &str) -> Self {
        let mut graph = Graph::new();
        let mut proto = Vec::new();
        File::open(model_path).unwrap().read_to_end(&mut proto);
        graph.import_graph_def(&proto, &ImportGraphDefOptions::new());
        let session = Session::new(&SessionOptions::new(), &graph).unwrap();
        Self { session, graph }
    }

    /// convert embeddings vector
    pub fn convert(&self, img_vec: Vec<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        // input tensor
        let x = Tensor::new(&[1, 224, 224, 3]).with_values(&img_vec)?;

        // Run the graph.
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.operation_by_name_required("resnet")?, 0, &x);
        let output = args.request_fetch(&self.graph.operation_by_name_required("Identity")?, 0);
        self.session.run(&mut args)?;

        // check result
        let output_tensor: Tensor<f32> = args.fetch(output)?;
        let output_array: Vec<f32> = output_tensor.iter().map(|x| x.clone()).collect();
        Ok(output_array)
    }

    /// convert embeddings vector from image path
    pub fn convert_from_img(&self, img_path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let img = ImageReader::open(img_path)?.decode()?;
        let resized_img = img.resize_exact(224 as u32, 224 as u32, FilterType::Lanczos3);
        let img_vec: Vec<f32> = resized_img.to_rgb8().to_vec().iter().map(|x| *x as f32).collect();
        self.convert(img_vec)
    }
}
