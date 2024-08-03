use burn::{
    nn::{Initializer, Linear, LinearConfig},
    prelude::*,
    LearningRate
};
use std::vec::Vec;
use super::PolicyGradientModel;

#[derive(Config, Debug)]
struct PolicyGradientModelConfig {
    num_inputs: usize,  // seed size
    num_outputs: usize, // number of mutators
    intermediate_layers_size: usize,
    num_layers: usize,
    epsilon_clip: f32,
    temperature: f32,
    random_percentage: f32,
    learning_rate: LearningRate,
    epochs: usize,
    batch_size: usize,
}

/// Implement ppo configurator 
impl PolicyGradientModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PolicyGradientModel <B> {
        let initializer = Initializer::Constant { value: 0.5 };
        // let initializer = Initializer::Constant{ value: 0.5 };
        let mut layers: Vec<Linear<B>> = Vec::new();
        // input layer
        layers.push(
            LinearConfig::new(
                self.num_inputs,
                self.intermediate_layers_size,
            ).with_initializer(initializer.clone()).init(device));
        // rest of the hidden layers
        for _ in 1..self.num_layers {
            layers.push(
                LinearConfig::new(
                    self.intermediate_layers_size,
                    self.intermediate_layers_size
                ).with_initializer(initializer.clone()).init(device));
        }
        // output layer (no need to clone initializer here)
        layers.push(
            LinearConfig::new(
                self.intermediate_layers_size,
                self.num_outputs
            ).with_initializer(initializer).init(device));
        
        let n_inputs = self.num_inputs;
        let n_outputs = self.num_outputs;
        PolicyGradientModel {
            layers,
            num_inputs: n_inputs,
            num_outputs: n_outputs,
            epsilon_clip: 0.5,
            temperature: 3.0,
            random_percentage: 0.75,
            learning_rate: 0.0001,
            num_epochs: 8,
            batch_size: 50,
        }
    }
}