use std::fmt;
use std::ops::{Mul};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Word(pub Vec<usize>);
impl Word {
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// `L(w, k)` means the largest numbers within the sum of the lengths of `k` disjoint (weakly) increasing sequences extracted from `w`
	#[allow(non_snake_case)]
	pub fn L(&self, k : usize) -> usize {
		self.to_tableau().shape().iter().cloned().take(k).sum()
	}
}
#[test] fn increasing_seq() {
	let word = Word(vec![1, 4, 2, 5, 6, 3]);
	assert_eq!(word.L(0), 0);
	assert_eq!(word.L(1), 4);
	assert_eq!(word.L(2), 6);
	assert_eq!(word.L(3), 6);
	assert_eq!(word.L(4), 6);
}
impl Mul for Word {
	type Output = Self;
    fn mul(mut self, mut rhs: Self) -> Self {
		self.0.append(&mut rhs.0);
		self
    }
}
#[test] fn mul_equivalence() {
	let lhs = Word(vec![1,2,3,4,2,3,4,9,2,3,2,3,4,2,2]);
	let rhs = Word(vec![1,2,3,42,343,464,334,33,2,3,2,5,3,43,2,1]);
	assert_eq!((lhs.clone() * rhs.clone()).to_tableau(), lhs.to_tableau() * rhs.to_tableau());
}

// ---------------------------------------------------------

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
struct Pair<T : PartialEq + Ord>(pub T, pub T);
#[test] fn pair_lexicographic_order() {
	assert!(Pair(1, 2) < Pair(2, 1));
	assert!(Pair(2, 1) < Pair(2, 2));
	assert!(Pair(2, 2) == Pair(2, 2));
}
impl<T : PartialEq + Ord> Pair<T> {
	pub fn from((index, value) : (T, T)) -> Pair<T> {
		Pair(index, value)
	}

	pub fn rev(self) -> Pair<T> {
		Pair(self.1, self.0)
	}
}

#[derive(Debug, Eq, Clone)]
pub struct TwoRowedArray(Vec<Pair<usize>>);
impl TwoRowedArray {
	pub fn new() -> TwoRowedArray {
		TwoRowedArray(Vec::new())
	}

	pub fn from_pairs(v : Vec<(usize, usize)>) -> TwoRowedArray {
		TwoRowedArray(v.into_iter().map(Pair::from).collect())
	}

	pub fn from_two_arrays(index_vec : Vec<usize>, value_vec : Vec<usize>) -> TwoRowedArray {
		if index_vec.len() != value_vec.len() {
			panic!("they cannot make up to an two rowed array")
		}
		TwoRowedArray::from_pairs(index_vec.into_iter().zip(value_vec.into_iter()).collect())
	}

	pub fn lexicographic_sort(&mut self) {
		self.0.sort()
	}
	pub fn lexicographic_ordered(&self) -> Self {
		let mut array = self.clone();
		array.lexicographic_sort();
		array
	}

	pub fn index_row(&self) -> Vec<usize> {
		self.0.iter().map(|p| {&p.0}).cloned().collect()
	}
		pub fn top_row(&self) -> Vec<usize> {self.index_row()}
	pub fn value_row(&self) -> Vec<usize> {
		self.0.iter().map(|p| {&p.1}).cloned().collect()
	}
		pub fn bottom_row(&self) -> Vec<usize> {self.index_row()}

	/// for the permutations, it is just the inverse
	pub fn inverse(&self) -> TwoRowedArray {
		TwoRowedArray(self.0.clone().into_iter().map(Pair::rev).collect())
	}

	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	pub fn push(&mut self, (index, value) : (usize, usize)) {
		self.0.push(Pair::from((index, value)));
	}

	pub fn iter(&self) -> impl Iterator<Item=(usize, usize)> + '_ {
		self.0.iter().cloned().map(|Pair(a, b)| (a, b))
	}
}
impl PartialEq for TwoRowedArray {
	fn eq(&self, other: &Self) -> bool {
        &self.lexicographic_ordered().0 == &other.lexicographic_ordered().0
    }
}
#[test] fn two_arrays() {
	assert_eq!(
		TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]),
		TwoRowedArray::from_pairs(
			vec![(1, 3),(2, 4),(3, 2),(4, 5),(2, 4),(4, 2),(4, 3),(2, 2)],
		)
	);
}
#[test] fn sort() {
	let array = TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]).lexicographic_ordered();
	assert_eq!(
		array.0,
		TwoRowedArray::from_two_arrays(
			vec![1,2,2,2,3,4,4,4],
			vec![3,2,4,4,2,2,3,5]
		).0
	);
}

impl fmt::Display for TwoRowedArray {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}\n{}",
			self.index_row().iter().fold(String::new(), |acc, e| {format!("{}{} ", acc, e)}),
			self.value_row().iter().fold(String::new(), |acc, e| {format!("{}{} ", acc, e)}),
		)
    }
}
#[test] fn display_array() {
	let array = TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]);
	assert_eq!(format!("{}", array), "1 2 3 4 2 4 4 2 \n3 4 2 5 4 2 3 2 ");
}
impl Word {
	pub fn to_two_rowed_array_0(&self) -> TwoRowedArray {
		TwoRowedArray::from_pairs(self.0.iter().cloned().enumerate().collect())
	}

	pub fn to_two_rowed_array_1(&self) -> TwoRowedArray {
		TwoRowedArray::from_pairs(self.0.iter().cloned().enumerate().map(|(index, value)| {(index + 1, value)}).collect())
	}
}