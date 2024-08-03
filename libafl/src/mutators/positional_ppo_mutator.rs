//! In this module we define the PosPPOMutator, a mutator that uses a PPO model to mutate an input
//! at a given position.a 
use std::vec::Vec;
use std::boxed::Box;

use burn::prelude::Backend;
use libafl_bolts::{rands::Rand, Named};
use libafl_bolts::tuples::IntoVec;

use crate::{
    mutators::Cow,
    prelude::{
        ppo_model::PolicyGradientModel,
        HasMaxSize,
        HasMutatorBytes,
        HasRand,
        MutationResult,
        Mutator,
        PosMutator
    },
    Error
};

use super::positional_mutations::positional_mutations;

#[derive(Debug)]
/// ['PosPPOMutator'] is the struct implementing a mutator that bases its mutation on a PPO
/// model, or PolicyGradient model if num_epochs == 1
pub struct PosPPOMutator <B, I, S>
where
    B: Backend,
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes
{
    position: usize,
    model: PolicyGradientModel<B>,
    mutations: Vec<Box<dyn PosMutator<I, S>>>,
}

impl <B, I, S> PosPPOMutator<B, I, S>
where
    B: Backend,
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes
{
    fn new(model: PolicyGradientModel<B>) -> Self {
        let mutations = positional_mutations();
        let mutations = mutations.into_vec();
        Self{
            position: 0,    // default value
            model,
            mutations,
        }
    }

    fn set_position (&mut self, pos: usize) {
        self.position = pos
    }
}

/// implement ppo traits and methods 
impl <B, I, S> Named for PosPPOMutator<B, I, S>
where
    B: Backend,
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes
{
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PolicyGradientModel");
        &NAME
    }
}

impl <B, I, S> Mutator<I, S> for PosPPOMutator<B, I, S>
where
    B: Backend,
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes
{
    fn mutate(&mut self, state: &mut S, input: &mut I) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            let input_len = input.len();
            let position = state.rand_mut().choose(0..input_len).unwrap();
            // let position = self.position;
            let (action, _probability, _entropy) = 
                self.model.get_action(
                    input.bytes_mut(),
                    state.max_size()
                );
            self.mutations[action].mutate(state, input, position).expect("mutation error in the PPO mutator");
            Ok(MutationResult::Mutated)
        }
    }
}