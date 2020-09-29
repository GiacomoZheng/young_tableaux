use std::cmp::PartialOrd;
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

pub fn replace_least_successor<T : PartialOrd>(x : T, v : &mut Vec<Option<T>>) -> Option<T> {
	for value in v.iter_mut().filter(|value| {value.is_some()}) {
		if value.as_ref() > Some(&x) {
			return value.replace(x);
		}
	}
	None
}
pub fn replace_greatest_predecessor<T : PartialOrd>(x : T, v : &mut Vec<Option<T>>) -> Option<T> {
	for value in v.iter_mut().filter(|value| {value.is_some()}).rev() {
		if value.as_ref() < Some(&x) {
			return value.replace(x);
		}
	}
	None
}