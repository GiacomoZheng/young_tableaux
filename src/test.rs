#[derive(Debug, Eq, Clone, Hash)]
pub struct TwoRowedArray(Vec<Pair>);
impl TwoRowedArray {
	fn from(v : Vec<Pair>) -> TwoRowedArray {
		TwoRowedArray(v.into_iter().collect())
	}

	pub fn from_pairs(v : Vec<(usize, usize)>) -> TwoRowedArray {
		TwoRowedArray::from(v.into_iter().map(Pair::from).collect())
	}

	pub fn from_two_arrays(index_vec : Vec<usize>, value_vec : Vec<usize>) -> TwoRowedArray {
		if index_vec.len() != value_vec.len() {
			panic!("they cannot make up an two rowed array")
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

	pub fn top_row(&self) -> Vec<usize> {
		self.0.iter().map(|p| {&p.0}).cloned().collect()
	}
	pub fn bottom_row(&self) -> Vec<usize> {
		self.0.iter().map(|p| {&p.1}).cloned().collect()
	}

	/// for the permutations, it is just the inverse
	pub fn inverse(self) -> TwoRowedArray {
		TwoRowedArray::from(self.0.into_iter().map(Pair::rev).collect())
	}

	// pub fn push(&mut self, (index, value) : (usize, usize)) {
	// 	self.0.push(Pair::from((index, value)));
	// }
}
#[test]
fn from_two_arrays() {
	assert_eq!(TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]), TwoRowedArray::from_pairs(
		vec![(1, 3),(2, 4),(3, 2),(4, 5),(2, 4),(4, 2),(4, 3),(2, 2)],
	));
}
#[test]
fn sort() {
	let mut array = TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]);
	array.lexicographic_sort();
	assert_eq!(array, TwoRowedArray::from_two_arrays(
		vec![1,2,2,2,3,4,4,4],
		vec![3,2,4,4,2,2,3,5]
	));
}

impl PartialEq for TwoRowedArray {
	fn eq(&self, other: &Self) -> bool {
        self.0.iter().collect::<HashSet<_>>() == other.0.iter().collect::<HashSet<_>>()
    }
}

impl fmt::Display for TwoRowedArray {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}\n{}",
			self.top_row().iter().fold(String::new(), |acc, e| {format!("{}{} ", acc, e)}),
			self.bottom_row().iter().fold(String::new(), |acc, e| {format!("{}{} ", acc, e)}),
		)
    }
}
#[test]
fn print_array() {
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

impl TwoRowedArray {
	#[allow(non_snake_case)]
	pub fn from_tableau_pair(mut P : Tableau, mut Q : Tableau) -> TwoRowedArray {
		let mut v = Vec::new();

		while let Some(row_index) = Q.greatest() {
			v.push((Q.pop_at(row_index), P.reverse_bumping(row_index)))
		}

		v.reverse();
		TwoRowedArray::from_pairs(v)
	}

	pub fn to_tableau_pair(&self) -> (Tableau, Tableau) {
		#[allow(non_snake_case)]
		let mut P = Tableau::new();
		#[allow(non_snake_case)]
		let mut Q = Tableau::new();
		for &Pair(index, value) in self.lexicographic_ordered().0.iter() {
			Q.place(P.row_bumping(value), index);
		}
		
		(P, Q)
	}
}
#[test]
fn conversion() {
	let array = TwoRowedArray::from_two_arrays(vec![3,4,5,2,2,1], vec![1,2,2,2,3,2]);
	#[allow(non_snake_case)]
	let (P, Q) = array.to_tableau_pair();
	// eprintln!("{}\n{}", P, Q);
	assert_eq!(TwoRowedArray::from_tableau_pair(P, Q), array);
}