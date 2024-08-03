//! PPO model implementation
use std::vec::Vec;
use burn::{prelude::{Backend, Data, Shape}, tensor::{backend::AutodiffBackend, Bool, Int, Tensor}};
use super::{
    PolicyGradientModel,
    train
};
pub use core::usize;

/// structure implementing the support memory. It is needed to store the batches for the PPO training  
#[derive(Debug)]
pub struct Memory {
    states: Vec<Vec<u8>>,          // input seeds
    actions: Vec<i32>,            // selected mutator's index
    probabilities: Vec<f32>,        // probability of selected mutator
    rewards: Vec<f32>,              // 1 if new coverage, 0 else
    size: usize,                    // memory (and batch) size
}

impl Memory {
    fn new(size: usize) -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            probabilities: Vec::new(),
            rewards: Vec::new(),            
            size,
        }
    }

    fn push(&mut self, state: Vec<u8>, action: i32, probability: f32, reward: f32) {
        self.states.push(state);
        self.actions.push(action);
        self.probabilities.push(probability);
        self.rewards.push(reward);
    }

    fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.probabilities.clear();
    }

    fn is_full(&self) -> bool {
        self.states.len() >= self.size
    }

    fn add_experience <B: AutodiffBackend> (
        &mut self,
        model: &mut PolicyGradientModel<B>,
        state: Vec<u8>,
        action: i32,
        probability: f32,
        reward: f32,
        // model: PolicyGradientModel<B>,
    ) {
        self.push(
            state,
            action,
            probability,
            reward,
        );
        if self.is_full() {
            train(model.clone(), self);
            self.clear();
        }
    }
    
    pub fn states(&self) -> &Vec<Vec<u8>> {
        &self.states
    }

    pub fn actions(&self) -> &Vec<i32> {
        &self.actions
    }

    pub fn probabilities(&self) -> &Vec<f32> {
        &self.probabilities
    }

    pub fn rewards(&self) -> &Vec<f32> {
        &self.rewards
    }

    pub fn size(&self) -> usize {
        self.size
    }

}

pub fn elementwise_min<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let rhs_lower: Tensor<B, D, Bool> = rhs.clone().lower(lhs.clone());
    lhs.clone().mask_where(rhs_lower, rhs.clone())
}


pub trait IntBatchifiable {
    fn to_tensor <B: Backend> (&self) -> Tensor<B,2, Int>;
}

impl IntBatchifiable for &Vec<i32> {
    fn to_tensor <B: Backend> (&self) -> Tensor<B,2, Int> {
        let device: <B as Backend>::Device = Default::default();
        Tensor::<B, 2, Int>::from_data(
            Data::new((*self).clone(), Shape::new([1, self.len()])).convert(),
            &device
        )        
    }
}

pub trait FpBatchifiable {
    fn to_tensor <B: Backend> (&self) -> Tensor<B,2>;
}



impl FpBatchifiable for &Vec<u8> {
    fn to_tensor <B: Backend> (&self) -> Tensor<B,2> {
        let device: <B as Backend>::Device = Default::default();
        Tensor::<B, 2>::from_data(
            Data::new((*self).clone(), Shape::new([1, self.len()])).convert(),
            &device
        )
    }
}

impl FpBatchifiable for &Vec<f32> {
    fn to_tensor <B: Backend> (&self) -> Tensor<B,2> {
        let device: <B as Backend>::Device = Default::default();
        Tensor::<B, 2>::from_data(
            Data::new((*self).clone(), Shape::new([1, self.len()])).convert(),
            &device
        )
    }
}


// Input requirement: the seeds musut be already padded
impl FpBatchifiable for &Vec<Vec<u8>> {
    fn to_tensor <B: Backend> (&self) -> Tensor<B, 2> {
        let device: <B as Backend>::Device = Default::default();
        let seed_inputs = self
            .iter()
            .map(|item|
                Tensor::<B, 2>::from_data(
                Data::new((*item).clone(), Shape::new([1, item.len()])).convert(),
                &device
                )
            )
            .collect();   
        let seed_inputs: Tensor<B, 2> = Tensor::cat(seed_inputs, 0).to_device(&device);
        seed_inputs
    }
}
