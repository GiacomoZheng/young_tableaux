pub mod order;

use std::ops::{Index, IndexMut};
use std::cmp::{PartialEq, PartialOrd, Ordering, max};

use std::iter::{repeat};
use std::fmt;

#[derive(Eq, Clone)]
pub struct VecTail<T : PartialEq> {
	heads : Vec<T>,
	tail : T,
	length : usize,
}
impl<T : PartialEq + Default> VecTail<T> {
	pub fn new() -> VecTail<T> {
		VecTail {
			heads : Vec::new(),
			tail : T::default(),
			length : 0,
		}
	}
}
impl<T : PartialEq> VecTail<T> {
	pub fn from(v : Vec<T>, tail : T) -> VecTail<T> {
		let mut vt = VecTail {
			length : v.len(),
			heads : v,
			tail,
		};

		vt.strip();

		vt
	}

	/// ? do I really need to pub it?
	pub fn strip(&mut self) {
		self.length = self.significant_length()
	}

	pub fn significant_length(&self) -> usize {
		let mut length = self.length;
		while length > 0 && self[length - 1] == self.tail {
			length -= 1;
		}
		length
	}

	pub fn is_empty(&self) -> bool {
		self.length == 0
	}

	fn push(&mut self, value: T) {
		if let Some(s) = self.heads.get_mut(self.length) {
			*s = value;
		} else {
			self.heads.push(value);
		}
		self.length += 1;
	}

	pub fn iter_finite(&self) -> std::iter::Take<std::slice::Iter<'_, T>> {
		self.heads.iter().take(self.significant_length())
	}

	pub fn iter(&self) -> std::iter::Chain<std::iter::Take<std::slice::Iter<'_, T>>, std::iter::Repeat<&T>> {
		self.iter_finite().chain(repeat(&self.tail))
	}

	pub fn iter_finite_mut(&mut self) -> std::iter::Take<std::slice::IterMut<'_, T>> {
		let length = self.significant_length();
		self.heads.iter_mut().take(length)
	}

	pub fn into_iter_finite(self) -> std::iter::Take<std::vec::IntoIter<T>> {
		let length = self.significant_length();
		self.heads.into_iter().take(length)
	}
}
#[test] fn length() {
	let v = VecTail::from(vec![1,2,3,4,0,0,0], 0);
	assert_eq!(v.length, 4);
}
impl<T : PartialOrd> VecTail<T> {
	pub fn is_weakly_decreasing(&self) -> Result<(), String> {
		order::is_weakly_decreasing(&self.iter_finite().collect())?;
		Ok(())
	}
}
impl<T : PartialEq> PartialEq for VecTail<T> {
	fn eq(&self, other: &Self) -> bool {
		self.iter().zip(other.iter()).take(max(self.length, other.length)).all(|(s, o)| {*s == *o})
    }
}
impl<T : Ord> PartialOrd for VecTail<T> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		let (mut less, mut equal, mut greater) = (0, 0, 0);
		for (s, o) in self.iter().zip(other.iter()).take(max(self.length, other.length)) {
			match s.cmp(o) {
				Ordering::Less => less += 1,
				Ordering::Greater => greater += 1,
				Ordering::Equal => {
					equal += 1;
					less += 1;
					greater += 1;
				}
			}
		}

		// println!("{},{},{},{}", equal, less, greater, self.length);

		if equal == self.length {
			Some(Ordering::Equal)
		} else if less == self.length {
			Some(Ordering::Less)
		} else if greater == self.length {
			Some(Ordering::Greater)
		} else {
			None
		}
    }
}
#[test] fn eq() {
	let inf_1 = VecTail::from(vec![1,2,3,4], 0);
	let inf_2 = VecTail {
		heads : vec![1,2,3,4, 0],
		tail : 0,
		length : 4,
	}; // just for test
	assert_eq!(inf_1, inf_2);

	let inf_1 = VecTail::from(vec![2,2,3,4], 0);
	let inf_2 = VecTail::from(vec![1,2,3,4], 0);
	assert_ne!(inf_1, inf_2);
	
	let inf_1 = VecTail::from(vec![
        vec![   None, Some(1), Some(2)],
        vec![   None],
        vec![Some(5)],
	], vec![]);
	let inf_2 = VecTail::from(vec![
        vec![   None, Some(1),    None],
        vec![   None],
        vec![Some(5)],
	], vec![]);
	assert_ne!(inf_1, inf_2);
}
#[test] fn ord() {
	let inf_1 = VecTail::from(vec![1,2,3,4], 0);
	let inf_2 = VecTail::from(vec![1,2,3,4], 0);
	let inf_3 = VecTail::from(vec![1,2,2,4], 0);
	assert!(inf_1 <= inf_2);
	assert!(inf_3 < inf_2);
}
impl<T : PartialEq + Copy> VecTail<Vec<T>> {
	pub fn pre_transpose(&self) -> VecTail<Vec<T>> {
		let mut v : Vec<Vec<T>> = Vec::new();
		for index in 0..self[0].len() {
			let mut tmp_v = Vec::new();
			for row in self.iter() {
				if index < row.len() {
					tmp_v.push(row[index]);
				} else {
					break;
				}
			}
			v.push(tmp_v);
		}
		VecTail::from(v, Vec::new())
	}
}
#[test] fn pre_transpose() {
	assert_eq!(
		VecTail::from(vec![
			vec![   None, Some(1), Some(2)],
			vec![   None, Some(2)],
			vec![Some(2)],
		], Vec::new()).pre_transpose(),
		VecTail::from(vec![
			vec![   None,   None, Some(2)],
			vec![Some(1), Some(2)],
			vec![Some(2)],
		], Vec::new())
	)
}
impl<T : PartialEq + fmt::Debug> fmt::Debug for VecTail<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "({}{:?}{})", self.iter_finite().fold(String::new(), |acc, e| {format!("{}{:?}, ", acc, e)}), self.tail, "...")
	}
}
#[test] fn display_vec() {
	let list = VecTail::from(vec![1,2,3], 0);
	// println!("{:?}", list);
	assert_eq!(String::from("(1, 2, 3, 0...)"), format!("{:?}", list) );
}

impl<T : PartialEq> Index<usize> for VecTail<T> {
	type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
		if index < self.length {
			self.heads.index(index)
		} else {
			&self.tail
		}
    }
}
// ! if we assign the tail to the last element, there might be some error
impl<T : PartialEq + Clone> IndexMut<usize> for VecTail<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		if index < self.length {
			self.heads.index_mut(index)
		} else {
			while self.length <= index {
				self.push(self.tail.clone());
			}
			self.heads.index_mut(index)
		} 
    }
}
#[test] fn index() {
	let mut list = VecTail::from(vec![1,2,3], 0);
	// println!("{:?}", list);
	assert_eq!(list[3], 0);
	assert_eq!(list[2], 3);

	list[1] = 2;
	assert_eq!(list[1], 2);
	list[3] = 1;
	assert_eq!(list[3], 1);
	assert_eq!(list.length, 4);
}
