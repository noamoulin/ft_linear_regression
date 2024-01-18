/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   trainer.rs                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: noa <noa@student.42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/01/18 13:18:17 by noa               #+#    #+#             */
/*   Updated: 2024/01/18 20:17:25 by noa              ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use std::{fs::File, error::Error, fmt};
use csv::{ReaderBuilder, StringRecord};
use plotters::prelude::*;

struct DataSet {
    x: Vec<f64>,
    y: Vec<f64>,
    x_name: String,
    y_name: String
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
        Ok(DataSet {x: x, y: y, x_name: labels[0].to_string(), y_name: labels[1].to_string()})
    }
}

impl<'a> Model<'a> {
    fn new(dataset: &DataSet) -> Model {
        Model {data: dataset, a: 0., b: 0.}
    }
    fn display(&self) {
        let name = self.data.y_name.clone() + " vs " + self.data.x_name.as_str();
        let file_name = name.clone().replace(" ", "_") + ".png";
        let data_x = &self.data.x;
        let data_y = &self.data.y;
        let (width, height) = (800, 600);
        let backend = BitMapBackend::new(file_name.as_str(), (width, height))
            .into_drawing_area();
        backend.fill(&WHITE).unwrap();
    
        let x_min = *data_x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let x_max = *data_x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_min = *data_y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_max = *data_y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
        let mut chart = ChartBuilder::on(&backend)
            .caption(name.clone().as_str(), ("helvetica", 20))
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

    fn train(&mut self, n_epochs: u32) {
        let learning_rate = 0.001;
        for _ in 0..n_epochs {
            let grad = self.error_gradient();
            self.a -= grad.0 * learning_rate;
            self.b -= grad.1 * learning_rate;
        }
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

fn normalize_vectors_together(vec1: &mut Vec<f64>, vec2: &mut Vec<f64>) {
    if vec1.is_empty() || vec2.is_empty() || vec1.len() != vec2.len() {
        return; // Les vecteurs doivent avoir la même taille et ne pas être vides
    }

    // Trouver les valeurs minimales et maximales combinées des deux vecteurs
    let min_val_combined = *vec1.iter().chain(vec2.iter()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_val_combined = *vec1.iter().chain(vec2.iter()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    // Normaliser chaque élément des deux vecteurs en utilisant les statistiques combinées
    for (val1, val2) in vec1.iter_mut().zip(vec2.iter_mut()) {
        *val1 = -1.0 + 2.0 * (*val1 - min_val_combined) / (max_val_combined - min_val_combined);
        *val2 = -1.0 + 2.0 * (*val2 - min_val_combined) / (max_val_combined - min_val_combined);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut dataset = DataSet::new("/home/noa/ft_linear_regression/data/data.csv")?;
    normalize_vectors_together(&mut dataset.x, &mut dataset.y);
    let mut model = Model::new(&dataset);

    model.train(10000);
    model.display();

    println!("{}, {}", model.a, model.b);
    Ok(())
}