/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   trainer.rs                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: nomoulin <nomoulin@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/01/18 13:18:17 by noa               #+#    #+#             */
/*   Updated: 2024/01/19 02:35:55 by nomoulin         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use std::{fs::File, error::Error, fmt, env};
use csv::{ReaderBuilder, StringRecord};
use plotters::prelude::*;

struct DataSet {
    x_or: Vec<f64>,
    y_or: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    x_name: String,
    y_name: String,
    max: f64
}
#[derive(Debug)]
enum DataSetError {
    NonNumericValue,
    InvalidColumnCount
}

impl fmt::Display for DataSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataSetError::InvalidColumnCount => write!(f, "Each record must have exactly two columns"),
            DataSetError::NonNumericValue => write!(f, "Each field must contain a numeric value"),
        }
    }
}

impl Error for DataSetError {}
struct Model<'a> {
    data: &'a DataSet,
    a: f64,
    b: f64
}

impl DataSet {
    fn new(path: &str) -> Result<DataSet, Box<dyn Error>> {
        let data_file = File::open(path)?;
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(data_file);

        let labels: Vec<String> = if let Some(line) = reader.records().nth(0) {
            let record: StringRecord = line?.clone();
            record.iter().map(|s| s.to_string()).collect()
        } else {
            return Err(Box::new(DataSetError::InvalidColumnCount));
        };
        if labels.len() != 2 {
            return Err(Box::new(DataSetError::InvalidColumnCount))
        };

        let mut x: Vec<f64> = vec![];
        let mut y: Vec<f64> = vec![];

        for record in reader.records() {
            let raw: StringRecord = record?;

            if raw.len() != 2 {
                return Err(Box::new(DataSetError::InvalidColumnCount));
            };
            x.push(raw[0].parse::<f64>().map_err(|_| DataSetError::NonNumericValue)?);
            y.push(raw[1].parse::<f64>().map_err(|_| DataSetError::NonNumericValue)?);
        }
        let (xn, yn, max) = normalized_vectors(&x, &y);
        Ok(DataSet {max: max, x_or: x, y_or: y, x: xn, y: yn, x_name: labels[0].to_string(), y_name: labels[1].to_string()})
    }
}

impl<'a> Model<'a> {
    fn new(dataset: &DataSet) -> Model {
        Model {data: dataset, a: 0., b: 0.}
    }
    fn plot(&self, path: &str, width: u32, height: u32) {
        let chart_name = self.data.y_name.clone() + " vs " + self.data.x_name.as_str();
        let data_x = &self.data.x_or;
        let data_y = &self.data.y_or;
        let backend = BitMapBackend::new(path, (width, height))
            .into_drawing_area();
        backend.fill(&WHITE).unwrap();
    
        let x_min = *data_x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let x_max = *data_x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_min = *data_y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_max = *data_y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
        let mut chart = ChartBuilder::on(&backend)
            .caption(chart_name, ("helvetica", 20))
            .x_label_area_size(40)
            .y_label_area_size(40)
            .margin(5)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();
    
        chart.draw_series(
            data_x.iter().zip(data_y.iter()).map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())),
        ).unwrap();

        chart.draw_series(LineSeries::new(
            [x_min, x_max].map(|x| (x as f64, self.a * x as f64 + self.b)),
            &BLUE,
        )).unwrap();

        chart.configure_mesh().x_desc(self.data.x_name.clone()).y_desc(self.data.y_name.clone()).draw().unwrap();
    }

    fn mean_error(&self, a: f64, b: f64) -> f64 {
        self.data.x.iter().zip(self.data.y.iter()).map(|(x, y)| (a * x + b - y).powf(2.)).sum::<f64>() / (self.data.x.len() as f64)
    }

    fn error_gradient(&self) -> (f64, f64) {
        nabla(|x, y| self.mean_error(x, y), self.a, self.b)
    }

    fn train(&mut self, d_rms: f64) {
        let learning_rate = 0.01;
        let mut previous_error = self.mean_error(self.a, self.b);
        loop {
            let grad = self.error_gradient();
            self.a -= grad.0 * learning_rate;
            self.b -= grad.1 * learning_rate;

            let error = self.mean_error(self.a, self.b);
            if previous_error - error < d_rms {
                break;
            }
            previous_error = error;
        }
        self.b *= self.data.max;
    }
}

fn nabla<F>(f: F, x: f64, y: f64) -> (f64, f64)
where
    F: Fn (f64, f64) -> f64,
{
    let h = 1e-10;
    
    let df_dx = (f(x + h, y) - f(x, y)) / h;
    let df_dy = (f(x, y + h) - f(x, y)) / h;

    (df_dx, df_dy)
}

fn normalized_vectors(a: &Vec<f64>, b: &Vec<f64>) -> (Vec<f64>, Vec<f64>, f64) {
    let maxabs_a = *a.iter().max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap()).unwrap();
    let maxabs_b = *b.iter().max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap()).unwrap();
    let max: f64 = if maxabs_a > maxabs_b {
        maxabs_a
    }
    else {
        maxabs_b
    };
    (a.into_iter().map(|elm| elm / max).collect(), b.into_iter().map(|elm| elm / max).collect(), max)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let dataset = DataSet::new(args[1].as_str())?;
    let width = args[3].parse::<u32>().unwrap();
    let height = args[4].parse::<u32>().unwrap();

    let mut model = Model::new(&dataset);
    model.train(1e-10);
    model.plot(args[2].as_str(), width, height);

    println!("y = {}x, + {}", model.a, model.b);
    Ok(())
}