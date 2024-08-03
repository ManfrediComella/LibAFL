//! PPO model implementation 
use std::vec::Vec;
use rand::{distributions::WeightedIndex, prelude::Distribution};
use rand::thread_rng;

use crate::mutators::ppo_model::memory::{
    elementwise_min,
    IntBatchifiable,
    FpBatchifiable,
    Memory
};

use num_traits::ToPrimitive;
use burn::{
    module::AutodiffModule,
    nn::Linear,
    grad_clipping::GradientClippingConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{
        activation::{softmax, tanh}, backend::AutodiffBackend, Int, Tensor,
    },
    LearningRate
};

mod memory;
mod model_config;

/// structure that actually encapsulates all the logic of the PPO model  
#[derive(Module, Debug)]
pub struct PolicyGradientModel <B: Backend> {
    layers: Vec<Linear<B>>,
    num_inputs: usize,      // max seed size
    num_outputs: usize,     // number of mutators
    epsilon_clip: f32,
    temperature: f32,
    random_percentage: f32,
    learning_rate: LearningRate,
    num_epochs: usize,
    batch_size: usize,
}

impl<B: Backend> PolicyGradientModel<B> {
    // # Shapes
    ///   - Input [batch_size, num_inputs] == [batch_size, seed_size] + positional information
    ///   - Output [batch_size, num_actions]
    fn forward(&self, input_seeds_batch: Tensor<B, 2>) -> Tensor<B, 2> {
        // Create a new tensor
        let mut x: Tensor<B, 2> = input_seeds_batch;
        // Forward pass through all but the last layer, with ReLU activation
        for i in 0..(self.layers.len() - 1) {
            x = self.layers[i].forward(x);
            x = tanh(x);
        }
        // Forward pass through the last layer, NO activation
        x = self.layers[self.layers.len() - 1].forward(x);
        // Apply softmax function to the output and get a probability distribution (heatmap)
        // log_softmax(x, 1) // [batch_size, output_map_size]
        softmax(x, 1) // [batch_size, output_map_size]
    }

    /// This method returns a HEATMAP, a probability distribution over the set of actions
    fn get_policy (
        &self,
        input: &[u8],
        max_state_size: usize,         // number of actions == number of possible positions to mutate == seed size
    ) -> (Vec<f32>, f32) {
        let device: <B as Backend>::Device = Default::default();
        // Add padding to input by setting exceeding bytes to 0
        let mut padded_input: Vec<f32> = input.iter().map(|el| *el as f32).collect();
        padded_input.resize(max_state_size, 0.0);
        // Feed the input to the model and get a probability distribution
        let input_tensor: Tensor<B, 2>  = Tensor::<B, 2>::from_data(
            Data::new(padded_input, Shape::new([1, max_state_size])).convert(), &device);
        let output_tensor: Tensor<B, 2> = self.forward(input_tensor);
        let entropy = -(output_tensor.clone().log() * output_tensor.clone())
            .sum_dim(1)
            .mean();
        let entropy = entropy.to_data().value[0].to_f32().expect("could not convert to f32");
        // Consider an amount of probabilities equal to the non-padded seed size == number of possible actions
        let mut probs: Vec<f32> = output_tensor.to_data().value.iter()
            .map(|element: &<B as Backend>::FloatElem| element.to_f32().expect("could not convert to f32"))
            .collect();
        probs.truncate(input.len());
        let sum: f32 = probs.iter().sum();
        // Return normalized probabilities and entropy
        if sum != 0.0 {
            // if sum != 0, then divide each probability by the total sum and return the probs and entropy
            (
                probs.iter()
                    .map(|element: &f32| element/sum)
                    .collect(),
                entropy
            )
        } else {
            // if for some reason the sum is null, return an uniform distribution and the entropy
            // this casting is ok up to 16,777,217 (2^24 + 1), more than enough 
            let den: f32 = input.len() as f32;
            let probs: Vec<f32> = vec![1.0 / den ; max_state_size];
            (probs, entropy)
        }
    }

    /// This method returns the index of the next action, together with the entropy value and the probability of said action
    pub fn get_action(&self, input: &[u8], max_input_size: usize) -> (usize, f32, f32) {
        let (probs, entropy) = self.get_policy(input,max_input_size);
    
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = thread_rng();
        let action = dist.sample(&mut rng);

        // actions.get_and_mutate(action.into(), state,input);

        (action, probs[action], entropy)
    }

    pub fn get_input_size(&self) -> usize {
        self.num_inputs
    }
}

/// Implement the training loop for the ppo model
pub fn train <B: AutodiffBackend> (
    mut ppo_model: PolicyGradientModel<B>,
    memory: &Memory,
) -> PolicyGradientModel<B> {
    let states: Tensor<B, 2> = memory.states().to_tensor();
    let advantages: Tensor<B, 2> = memory.rewards().to_tensor();
    let old_policies: Tensor<B, 2> = ppo_model.forward(states.clone());
    let actions: Tensor<B, 2, Int> = memory.actions().to_tensor().reshape([memory.size(), 1]);
    
    for _ in 0..10 {
        let state_batch: Tensor<B, 2> = states.clone();
        let action_batch: Tensor<B, 2, Int> = actions.clone();
        let advantage_batch: Tensor<B, 2> = advantages.clone();
        let policy_batch: Tensor<B, 2> = ppo_model.forward(state_batch);
        
        let ratios: Tensor<B, 2> = policy_batch
            .clone()
            .div(old_policies.clone())
            .gather(1, action_batch)
            .reshape([1, -1]);
        
        let clipped_ratios: Tensor<B, 2> = ratios
            .clone()
            .clamp(1.0 - 0.2, 1.0 + 0.2);
        
        let loss: Tensor<B, 1> = -elementwise_min(
            ratios * advantage_batch.clone(),
            clipped_ratios * advantage_batch,
        )
        .sum();
            
        let policy_negative_entropy: Tensor<B, 1> = -(policy_batch.clone().log() * policy_batch)
            .sum_dim(1)
            .mean();
        println!("policy_neg_entropy: {}", policy_negative_entropy);

        let loss = loss + policy_negative_entropy.mul_scalar(0.4);
        
        ppo_model = update_parameters(
            ppo_model.learning_rate,
            loss,
            ppo_model,
            &mut AdamWConfig::new()
                .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
                .init(),
        );
    }
    ppo_model
}

fn update_parameters <B: AutodiffBackend, M: AutodiffModule<B>> (
    learning_rate: LearningRate,
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}