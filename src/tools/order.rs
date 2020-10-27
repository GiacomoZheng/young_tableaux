use std::cmp::{PartialOrd, Ordering};

pub fn is_weakly_decreasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
	if !v.is_empty() {
		for index in 0..(v.len() - 1) {
			if v[index + 1] > v[index] {
				return Err("should be weakly deceasing");
			}
		}
	}
	Ok(())
}
pub fn is_weakly_increasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
	if !v.is_empty() {
		for index in 0..(v.len() - 1) {
			if v[index + 1] < v[index] {
				return Err("should be weakly inceasing");
			}
		}
	}
	Ok(())
}

#[allow(dead_code)]
pub fn is_strictly_decreasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
	if !v.is_empty() {
		for index in 0..(v.len() - 1) {
			if v[index + 1] >= v[index] {
				return Err("should be strictly deceasing");
			}
		}
	}
	Ok(())
}
pub fn is_strictly_increasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
	if !v.is_empty() {
		for index in 0..(v.len() - 1) {
			if v[index + 1] <= v[index] {
				return Err("should be strictly inceasing");
			}
		}
	}
	Ok(())
}

pub fn replace_least_successor<T : PartialOrd + Copy>(x : T, v : &mut Vec<T>) -> Option<T> {
	for value in v.iter_mut() {
		if *value > x {
			let res = Some(*value);
			*value = x;
			return res;
		}
	}
	None
}

pub fn replace_greatest_predecessor<T : PartialOrd + Copy>(x : T, v : &mut Vec<T>) -> Option<T> {
	for value in v.iter_mut().rev() {
		if *value < x {
			let res = Some(*value);
			*value = x;
			return res;
		}
	}
	None
}

use std::ops::Range;
/// ? I'm not sure about that
pub fn compare<Idx : PartialOrd>(range : &Range<Idx>, k : Idx) -> Ordering {
	if k < range.start {
		Ordering::Less
	} else if k >= range.end {
		Ordering::Greater
	} else {
		Ordering::Equal
	}
}