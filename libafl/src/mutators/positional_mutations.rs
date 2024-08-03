//! A wide variety of mutations that given a certain position (usize) apply some actions on the input seed.
//! Used during fuzzing. 

use libafl_bolts::{
    rands::Rand,
    Named,
    prelude::tuple_list
};

use alloc::borrow::Cow;
use tuple_list::tuple_list_type; 
// alloc::borrowToOwned,
//  vec::Vec,
use core::ops::Range;
use core::mem::size_of;
use crate::{
    
    inputs::HasMutatorBytes,
    // corpus::Corpus,
    mutators::{
        PosMutator,
        MutationResult,
        mutations::{
            buffer_copy, buffer_self_copy, buffer_set, rand_range,
            INTERESTING_8, INTERESTING_16, INTERESTING_32, ARITH_MAX,
        },
    },
    state::{HasRand, HasMaxSize,},
    Error,
};
use std::cmp::min;
use std::vec::Vec;

/// Generate a range of index of random length starting from a specified position.
/// upper represents the upper bound for the range, max_len is the maximal extension of said range.
/// Requirement: position is a generic value between 0 and upper.
#[inline]
pub fn pos_rand_range<S: HasRand>(state: &mut S, upper: usize, max_len: usize, position: usize) -> Range<usize> {
    let len = 1 + state.rand_mut().below(max_len);
    
    let offset1 = position;
    let mut offset2 = offset1 + len;
    if offset2 > upper {
        offset2 = upper;
    }
    offset1..offset2
}


/// Tuple type of the mutations that can be selected by a positional mutator
pub type PosMutationsType = tuple_list_type!(
    PosBitFlipMutator,
    PosByteFlipMutator,
    PosByteIncMutator,
    PosByteDecMutator,
    PosByteNegMutator,
    PosByteRandMutator,
    PosByteAddMutator,
    PosWordAddMutator,
    PosDwordAddMutator,
    PosQwordAddMutator,
    PosByteInterestingMutator,
    PosWordInterestingMutator,
    PosDwordInterestingMutator,
    PosBytesDeleteMutator,
    PosBytesDeleteMutator,
    PosBytesDeleteMutator,
    PosBytesDeleteMutator,
    PosBytesExpandMutator,
    PosBytesInsertMutator,
    PosBytesRandInsertMutator,
    PosBytesSetMutator,
    PosBytesRandSetMutator,
    PosBytesCopyMutator,
    PosBytesInsertCopyMutator,
    PosBytesSwapMutator,
);

/// Get the mutations that can be selected by a positional mutator
pub fn positional_mutations() -> PosMutationsType {
    tuple_list! (
        PosBitFlipMutator::new(),
        PosByteFlipMutator::new(),
        PosByteIncMutator::new(),
        PosByteDecMutator::new(),
        PosByteNegMutator::new(),
        PosByteRandMutator::new(),
        PosByteAddMutator::new(),
        PosWordAddMutator::new(),
        PosDwordAddMutator::new(),
        PosQwordAddMutator::new(),
        PosByteInterestingMutator::new(),
        PosWordInterestingMutator::new(),
        PosDwordInterestingMutator::new(),
        PosBytesDeleteMutator::new(),
        PosBytesDeleteMutator::new(),
        PosBytesDeleteMutator::new(),
        PosBytesDeleteMutator::new(),
        PosBytesExpandMutator::new(),
        PosBytesInsertMutator::new(),
        PosBytesRandInsertMutator::new(),
        PosBytesSetMutator::new(),
        PosBytesRandSetMutator::new(),
        PosBytesCopyMutator::new(),
        PosBytesInsertCopyMutator::new(),
        PosBytesSwapMutator::new(),
    )
}

macro_rules! pos_add_mutator_impl {
    ($name: ident, $size: ty) => {
        #[doc = concat!("Adds or subtracts a random value up to `ARITH_MAX` to a [`", stringify!($size), "`] at a specified position in the [`Vec`], in random byte order.")]
        #[derive(Default, Debug)]
        pub struct $name;

        #[allow(trivial_numeric_casts)]
        impl<I, S> PosMutator<I, S> for $name
        where
            S: HasRand,
            I: HasMutatorBytes,
        {
            fn mutate(
                &mut self,
                state: &mut S,
                input: &mut I,
                position: usize
            ) -> Result<MutationResult, Error> {
                if input.bytes().len() < size_of::<$size>() {
                    Ok(MutationResult::Skipped)
                } else {
                    // choose a random window of bytes (windows overlap) and convert to $size
                    let offset1 = position;
                    let mut offset2 = position + size_of::<$size>();
                    if offset2 > input.len() {                        
                        offset2 = input.len() - size_of::<$size>();
                    }
                    let bytes = & input.bytes()[offset1..offset2];
                    
                    let val = <$size>::from_ne_bytes(bytes.try_into().unwrap());

                    // mutate
                    let num = 1 + state.rand_mut().below(ARITH_MAX) as $size;
                    let new_val = match state.rand_mut().below(4) {
                        0 => val.wrapping_add(num),
                        1 => val.wrapping_sub(num),
                        2 => val.swap_bytes().wrapping_add(num).swap_bytes(),
                        _ => val.swap_bytes().wrapping_sub(num).swap_bytes(),
                    };

                    // set bytes to mutated value
                    let new_bytes = &mut input.bytes_mut()[offset1..offset2];
                    new_bytes.copy_from_slice(&new_val.to_ne_bytes());
                    Ok(MutationResult::Mutated)
                }
            }
        }

        impl Named for $name {
            fn name(&self) -> &Cow<'static, str> {
                static NAME: Cow<'static, str> = Cow::Borrowed(stringify!($name));
                &NAME
            }
        }

        impl $name {
            #[doc = concat!("Creates a new [`", stringify!($name), "`].")]
            #[must_use]
            pub fn new() -> Self {
                Self
            }
        }
    };
}

pos_add_mutator_impl!(PosByteAddMutator, u8);
pos_add_mutator_impl!(PosWordAddMutator, u16);
pos_add_mutator_impl!(PosDwordAddMutator, u32);
pos_add_mutator_impl!(PosQwordAddMutator, u64);

macro_rules! pos_interesting_mutator_impl {
    ($name: ident, $size: ty, $interesting: ident) => {
        /// Inserts an interesting value at a random place in the input vector
        #[derive(Default, Debug)]
        pub struct $name;

        impl<I, S> PosMutator<I, S> for $name
        where
            S: HasRand,
            I: HasMutatorBytes,
        {
            #[allow(clippy::cast_sign_loss)]
            fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
                if input.bytes().len() < size_of::<$size>() {
                    Ok(MutationResult::Skipped)
                } else {
                    let bytes = input.bytes_mut();
                    let upper_bound = (bytes.len() + 1 - size_of::<$size>());
                    // in case the position to mutate exceeds the upper bound, simply mutate the last possible byte
                    let idx = min(upper_bound, position);
                    // random choice of interesting val
                    let val = *state.rand_mut().choose(&$interesting).unwrap() as $size;
                    let new_bytes = match state.rand_mut().choose(&[0, 1]).unwrap() {
                        0 => val.to_be_bytes(),
                        _ => val.to_le_bytes(),
                    };
                    bytes[idx..idx + size_of::<$size>()].copy_from_slice(&new_bytes);
                    Ok(MutationResult::Mutated)
                }
            }
        }

        impl Named for $name {
            fn name(&self) -> &Cow<'static, str> {
                static NAME: Cow<'static, str> = Cow::Borrowed(stringify!($name));
                &NAME
            }
        }

        impl $name {
            #[doc = concat!("Creates a new [`", stringify!($name), "`].")]
            #[must_use]
            pub fn new() -> Self {
                Self
            }
        }
    };
}

pos_interesting_mutator_impl!(PosByteInterestingMutator, u8, INTERESTING_8);
pos_interesting_mutator_impl!(PosWordInterestingMutator, u16, INTERESTING_16);
pos_interesting_mutator_impl!(PosDwordInterestingMutator, u32, INTERESTING_32);


/// Bitflip mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBitFlipMutator;

impl<I, S> PosMutator<I, S> for PosBitFlipMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            let bit = 1 << state.rand_mut().choose(0..8).unwrap();
            input.bytes_mut()[position] ^= bit;
            /* // alternative
            let byte = &mut input.bytes_mut()[position];
            *byte ^= bit;
            */
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for PosBitFlipMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBitFlipMutator");
        &NAME
    }
}

impl PosBitFlipMutator {
    /// Creates a new [`BitFlipMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Byteflip mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosByteFlipMutator;

impl<I, S> PosMutator<I, S> for PosByteFlipMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, _state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            input.bytes_mut()[position] ^= 0xff;
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for PosByteFlipMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosByteFlipMutator");
        &NAME
    }
}

impl PosByteFlipMutator {
    /// Creates a new [`ByteFlipMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Byte increment mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosByteIncMutator ;

impl<I, S> PosMutator<I, S> for PosByteIncMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, _state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            input.bytes_mut()[position] += 1;
            Ok(MutationResult::Mutated)
        }
    }
}


impl Named for PosByteIncMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosByteIncMutator");
        &NAME
    }
}

impl PosByteIncMutator {
    /// Creates a new [`ByteIncMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Byte decrement mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosByteDecMutator;

impl<I, S> PosMutator<I, S> for PosByteDecMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, _state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            input.bytes_mut()[position] += 1;
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for PosByteDecMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosByteIncMutator");
        &NAME
    }
}

impl PosByteDecMutator {
    /// Creates a new [`ByteIncMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Byte negate mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosByteNegMutator;

impl<I, S> PosMutator <I, S> for PosByteNegMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, _state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            input.bytes_mut()[position] = !input.bytes_mut()[position] + 1;
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for PosByteNegMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosByteNegMutator");
        &NAME
    }
}

impl PosByteNegMutator {
    /// Creates a new [`ByteNegMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Byte random mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosByteRandMutator;

impl<I, S> PosMutator<I, S> for PosByteRandMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        if input.bytes().is_empty() {
            Ok(MutationResult::Skipped)
        } else {
            input.bytes_mut()[position] ^= 1 + state.rand_mut().below(254) as u8;
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for PosByteRandMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosByteRandMutator");
        &NAME
    }
}

impl PosByteRandMutator {
    /// Creates a new [`ByteRandMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}


/// Bytes delete mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesDeleteMutator;

impl<I, S> PosMutator<I, S> for PosBytesDeleteMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size <= 2 {
            return Ok(MutationResult::Skipped);
        }

        let range = pos_rand_range(state, size, size - 1, position);

        input.drain(range);

        Ok(MutationResult::Mutated)
    }
}

impl Named for PosBytesDeleteMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesDeleteMutator");
        &NAME
    }
}

impl PosBytesDeleteMutator {
    /// Creates a new [`BytesDeleteMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Bytes expand mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesExpandMutator;

impl<I, S> PosMutator<I, S> for PosBytesExpandMutator
where
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let max_size = state.max_size();
        let size = input.bytes().len();
        if size == 0 || size >= max_size {
            return Ok(MutationResult::Skipped);
        }

        let range = pos_rand_range(
                state,
                size,
                min(16, max_size - size),
                position
            );

        input.resize(size + range.len(), 0);
        unsafe {
            buffer_self_copy(
                input.bytes_mut(),
                range.start,
                range.start + range.len(),
                size - range.start,
            );
        }

        Ok(MutationResult::Mutated)
    }
}


impl Named for PosBytesExpandMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesExpandMutator");
        &NAME
    }
}

impl PosBytesExpandMutator {
    /// Creates a new [`BytesExpandMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Bytes insert mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesInsertMutator;

impl<I, S> PosMutator<I, S> for PosBytesInsertMutator
where
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let max_size = state.max_size();
        let size = input.bytes().len();
        if size == 0 || size >= max_size {
            return Ok(MutationResult::Skipped);
        }

        let mut amount = 1 + state.rand_mut().below(16);
        let offset = position;

        if size + amount > max_size {
            if max_size > size {
                amount = max_size - size;
            } else {
                return Ok(MutationResult::Skipped);
            }
        }

        let val = input.bytes()[state.rand_mut().below(size)];

        input.resize(size + amount, 0);
        unsafe {
            buffer_self_copy(
                input.bytes_mut(),
                offset,
                offset + amount,
                size - offset
            );
        }
        buffer_set(input.bytes_mut(), offset, amount, val);

        Ok(MutationResult::Mutated)
    }
}


impl Named for PosBytesInsertMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesInsertMutator");
        &NAME
    }
}

impl PosBytesInsertMutator {
    /// Creates a new [`BytesInsertMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}


/// Bytes random insert mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesRandInsertMutator;

impl<I, S> PosMutator<I, S> for PosBytesRandInsertMutator
where
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let max_size = state.max_size();    // max seed size
        let size = input.bytes().len();     // seed length
        if size >= max_size {
            return Ok(MutationResult::Skipped);
        }
        // amount of bytes to insert (1 to 16)
        let mut amount = 1 + state.rand_mut().below(16);
        // where to insert said bytes
        let offset = position;

        // ensure that the amount of bytes does not exceed the maximum seed size
        if size + amount > max_size {
            if max_size > size {
                amount = max_size - size;
            } else {
                return Ok(MutationResult::Skipped);
            }
        }
        // random value to store in the newly freshly inserted bytes
        let val = state.rand_mut().next() as u8;
        
        // resize the buffer to store more bytes
        input.resize(size + amount, 0);
        // input: PREF | SUFF ---> PREF | amount | SUFF 
        unsafe {
            buffer_self_copy(
                input.bytes_mut(),
                offset,
                offset + amount,
                size - offset
            );
        }
        buffer_set(input.bytes_mut(), offset, amount, val);

        Ok(MutationResult::Mutated)
    }
}


impl Named for PosBytesRandInsertMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesRandInsertMutator");
        &NAME
    }
}

impl PosBytesRandInsertMutator {
    /// Create a new [`BytesRandInsertMutator`]
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Bytes set mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesSetMutator;

impl<I, S> PosMutator<I, S> for PosBytesSetMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size == 0 {
            return Ok(MutationResult::Skipped);
        }
        let range = pos_rand_range(
                state,
                size,
                min(size, 16),
                position
            );
        let val = *state.rand_mut().choose(input.bytes()).unwrap();
        let quantity = range.len();
        buffer_set(input.bytes_mut(), range.start, quantity, val);

        Ok(MutationResult::Mutated)
    }
}

impl Named for PosBytesSetMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesSetMutator");
        &NAME
    }
}

impl PosBytesSetMutator {
    /// Creates a new [`BytesSetMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Bytes random set mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesRandSetMutator;

impl<I, S> PosMutator<I, S> for PosBytesRandSetMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size == 0 {
            return Ok(MutationResult::Skipped);
        }
        // default random value to insert
        let val = state.rand_mut().next() as u8;
        
        let mut quantity = min(size, 16);
        if position + quantity > size {
            quantity = size - position;
        }
        buffer_set(input.bytes_mut(), position, quantity, val);

        Ok(MutationResult::Mutated)
    }
}

impl Named for PosBytesRandSetMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesRandSetMutator");
        &NAME
    }
}

impl PosBytesRandSetMutator {
    /// Creates a new [`BytesRandSetMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Bytes copy mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesCopyMutator;

impl<I, S> PosMutator<I, S> for PosBytesCopyMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size <= 1 {
            return Ok(MutationResult::Skipped);
        }

        let range = rand_range(state, size, size - position);

        unsafe {
            buffer_self_copy(input.bytes_mut(), range.start, position, range.len());
        }

        Ok(MutationResult::Mutated)
    }
}


impl Named for PosBytesCopyMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesCopyMutator");
        &NAME
    }
}

impl PosBytesCopyMutator {
    /// Creates a new [`BytesCopyMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Bytes insert and self copy mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesInsertCopyMutator {
    tmp_buf: Vec<u8>,
}

impl<I, S> PosMutator<I, S> for PosBytesInsertCopyMutator
where
    S: HasRand + HasMaxSize,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size <= 1 || size >= state.max_size() {
            return Ok(MutationResult::Skipped);
        }
        // make sure that the sampled range is both in bounds and of an acceptable size
        let max_insert_len = min(size - position, state.max_size() - size);
        let range = pos_rand_range(
                state,
                size,
                min(16, max_insert_len),
                position
            );
        input.resize(size + range.len(), 0);
        self.tmp_buf.resize(range.len(), 0);
        unsafe {
            buffer_copy(
                &mut self.tmp_buf,
                input.bytes(),
                range.start,
                0,
                range.len(),
            );

            buffer_self_copy(
                input.bytes_mut(),
                position,
                position + range.len(),
                size - position,
            );
            buffer_copy(input.bytes_mut(), &self.tmp_buf, 0, position, range.len());
        }
        Ok(MutationResult::Mutated)
    }
}



impl Named for PosBytesInsertCopyMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesInsertCopyMutator");
        &NAME
    }
}

impl PosBytesInsertCopyMutator {
    /// Creates a new [`BytesInsertCopyMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Bytes swap mutation for inputs with a bytes vector
#[derive(Default, Debug)]
pub struct PosBytesSwapMutator {
    tmp_buf: Vec<u8>,
}

#[allow(clippy::too_many_lines)]
impl<I, S> PosMutator<I, S> for PosBytesSwapMutator
where
    S: HasRand,
    I: HasMutatorBytes,
{
    fn mutate(&mut self, state: &mut S, input: &mut I, position: usize) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size <= 1 {
            return Ok(MutationResult::Skipped);
        }

        let first = pos_rand_range(state, size, size, position);
        if state.rand_mut().next() & 1 == 0 && first.start != 0 {
            // The second range comes before first.

            let second = rand_range(state, first.start, first.start);
            self.tmp_buf.resize(first.len(), 0);
            unsafe {
                // If range first is larger
                if first.len() >= second.len() {
                    let diff_in_size = first.len() - second.len();

                    // copy first range to tmp
                    buffer_copy(
                        &mut self.tmp_buf,
                        input.bytes(),
                        first.start,
                        0,
                        first.len(),
                    );

                    // adjust second.end..first.start, move them by diff_in_size to the right
                    buffer_self_copy(
                        input.bytes_mut(),
                        second.end,
                        second.end + diff_in_size,
                        first.start - second.end,
                    );

                    // copy second to where first was
                    buffer_self_copy(
                        input.bytes_mut(),
                        second.start,
                        first.start + diff_in_size,
                        second.len(),
                    );

                    // copy first back
                    buffer_copy(
                        input.bytes_mut(),
                        &self.tmp_buf,
                        0,
                        second.start,
                        first.len(),
                    );
                } else {
                    let diff_in_size = second.len() - first.len();

                    // copy first range to tmp
                    buffer_copy(
                        &mut self.tmp_buf,
                        input.bytes(),
                        first.start,
                        0,
                        first.len(),
                    );

                    // adjust second.end..first.start, move them by diff_in_size to the left
                    buffer_self_copy(
                        input.bytes_mut(),
                        second.end,
                        second.end - diff_in_size,
                        first.start - second.end,
                    );

                    // copy second to where first was
                    buffer_self_copy(
                        input.bytes_mut(),
                        second.start,
                        first.start - diff_in_size,
                        second.len(),
                    );

                    // copy first back
                    buffer_copy(
                        input.bytes_mut(),
                        &self.tmp_buf,
                        0,
                        second.start,
                        first.len(),
                    );
                }
            }
            Ok(MutationResult::Mutated)
        } else if first.end != size {
            // The first range comes before the second range
            let mut second = rand_range(state, size - first.end, size - first.end);
            second.start += first.end;
            second.end += first.end;

            self.tmp_buf.resize(second.len(), 0);
            unsafe {
                if second.len() >= first.len() {
                    let diff_in_size = second.len() - first.len();
                    // copy second range to tmp
                    buffer_copy(
                        &mut self.tmp_buf,
                        input.bytes(),
                        second.start,
                        0,
                        second.len(),
                    );

                    // adjust first.end..second.start, move them by diff_in_size to the right
                    buffer_self_copy(
                        input.bytes_mut(),
                        first.end,
                        first.end + diff_in_size,
                        second.start - first.end,
                    );

                    // copy first to where second was
                    buffer_self_copy(
                        input.bytes_mut(),
                        first.start,
                        second.start + diff_in_size,
                        first.len(),
                    );

                    // copy second back
                    buffer_copy(
                        input.bytes_mut(),
                        &self.tmp_buf,
                        0,
                        first.start,
                        second.len(),
                    );
                } else {
                    let diff_in_size = first.len() - second.len();
                    // copy second range to tmp
                    buffer_copy(
                        &mut self.tmp_buf,
                        input.bytes(),
                        second.start,
                        0,
                        second.len(),
                    );

                    // adjust first.end..second.start, move them by diff_in_size to the left
                    buffer_self_copy(
                        input.bytes_mut(),
                        first.end,
                        first.end - diff_in_size,
                        second.start - first.end,
                    );

                    // copy first to where second was
                    buffer_self_copy(
                        input.bytes_mut(),
                        first.start,
                        second.start - diff_in_size,
                        first.len(),
                    );

                    // copy second back
                    buffer_copy(
                        input.bytes_mut(),
                        &self.tmp_buf,
                        0,
                        first.start,
                        second.len(),
                    );
                }
            }

            Ok(MutationResult::Mutated)
        } else {
            Ok(MutationResult::Skipped)
        }
    }
}


impl Named for PosBytesSwapMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("PosBytesSwapMutator");
        &NAME
    }
}

impl PosBytesSwapMutator {
    /// Creates a new [`BytesSwapMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}