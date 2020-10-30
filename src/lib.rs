use std::fmt;
use std::cmp::{PartialEq, PartialOrd, Ordering};
use std::ops::{Mul}; //, Deref, DerefMut};

pub trait MathClass {
	/// criteria for a math concept
	fn check(&self) -> Result<(), String>;
}

mod tableau;
pub use tableau::{Filling, Numbering, Tableau, StandardTableau, SkewTableau};

mod word;
pub use word::{Word, TwoRowedArray};

impl Mul for Tableau {
	type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
		for &r in rhs.to_word().0.iter() {
			self.row_bumping(r);
		}
		self
    }
}
#[test] fn mul_tableau() {
	let lhs = Word(vec![1,2,3,4,3]).to_tableau();
	let rhs = Word(vec![1,2,3,4,3]).to_tableau();
	assert_eq!((lhs * rhs).to_word(), Word(vec![4,2,3,4,1,1,2,3,3,3]));
	
	let lhs = Word(vec![1,2,3,4,3]).to_tableau();
	let rhs = Word(vec![3,6,3,4]).to_tableau();
	assert_eq!((lhs * rhs).to_word(), Word(vec![4,6,1,2,3,3,3,3,4]));
}

impl Word {
	pub fn to_skew_tableau(&self) -> SkewTableau {
		// + scan
		let mut vec : Vec<Vec<Option<usize>>> = Vec::new();
		let mut spaces : usize = 0;
		let mut iter = self.0.iter().peekable();

		let mut items : Option<Vec<Option<usize>>> = Some((0..spaces).map(|_| None).collect());
		while let Some(&left) = iter.next() {
			items.as_mut().unwrap().push(Some(left));
			if let Some(&&right) = iter.peek() {
				if left > right {
					spaces = items.as_ref().unwrap().len();
					vec.insert(0, items.take().unwrap());
					items = Some((0..spaces).map(|_| None).collect());
				}
			}
		}
		vec.insert(0, items.take().unwrap());
		SkewTableau::from(vec)
	}

	/// row insert one by one
	pub fn to_tableau(&self) -> Tableau {
		let mut tableau = Tableau::new();
		for &letter in self.0.iter() {
			tableau.row_insert(letter);
		}
		tableau
	}
}
impl Filling {
	/// w(T) = from left below side to right and row by row upwards
	pub fn to_word(&self) -> Word {
		Word(self.iter_finite().cloned().rev().flatten().collect())
	}
	// + to col word, page. 27
	/// w_col(T) = from left below side to upwards and col by col rightwards
	pub fn to_col_word(&self) -> Word {
		Word(self.transpose().iter_finite().cloned().rev().flatten().rev().collect())
	}
}

impl SkewTableau {
	/// w(T) = from left below side to right and row by row upwards
	pub fn to_word(&self) -> Word {
		Word(self.iter_finite().cloned().rev().flatten().filter(|e| {e.is_some()}).map(|e| {e.unwrap()}).collect())
	}
	// + to col word, page. 27
	/// w_col(T) = from left below side to upwards and col by col rightwards
	pub fn to_col_word(&self) -> Word {
		Word(self.pre_transpose().iter_finite().cloned().rev().flatten().filter(|e| {e.is_some()}).map(|e| {e.unwrap()}).rev().collect())
	}

}
#[test] fn word() {
	let tableau = SkewTableau::from(vec![
		vec![   None,Some(4),Some(5),Some(6),Some(7)],
		vec![Some(3),Some(6),Some(7)],
		vec![Some(7)],
	]);
	
	// println!("{}", tableau);
	let word = tableau.to_word();
	assert_eq!(word, Word(vec![7, 3, 6, 7, 4, 5, 6, 7]));

	let col_word = tableau.to_col_word();
	assert_eq!(col_word, Word(vec![7, 3, 6, 4, 7, 5, 6, 7]));
	
	let skew_tableau = word.to_skew_tableau();
	// println!("{}", skew_tableau);
	assert_eq!(skew_tableau.to_word(), word);

	assert_eq!(word.to_tableau().to_word(), Word(vec![7, 6, 7, 3, 4, 5, 6, 7]));

	assert_eq!(word.to_tableau().to_word(), col_word.to_tableau().to_word());
}
impl PartialOrd for Numbering { // +
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if let Some(order) = self.shape().partial_cmp(&other.shape()) {
			if order == Ordering::Equal {
				unimplemented!()
				// for i in (1..=self.shape().n()).rev() {
				// 	// self.to_col_word()
				// }
			} else {
				Some(order)
			}
		} else {
			None
		}
    }
}
#[test] fn order_numbering() {
	unimplemented!()
}

// --------------------------------------------------------------
#[derive(Debug, PartialEq, Eq)]
pub struct TableauPair(Tableau, Tableau);
impl MathClass for TableauPair {
	fn check(&self) -> Result<(), String> {
		if self.0.shape() != self.1.shape() {
			Err("these two tableau has different shape".into())
		} else {
			Ok(())
		}
	}
}
impl fmt::Display for TableauPair {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "P:\n{}\nQ:\n{}", self.value_tableau(), self.index_tableau())
    }
}

impl TableauPair {
	#[allow(non_snake_case)]
	pub fn from(P : Tableau, Q : Tableau) -> TableauPair {
		let T = TableauPair(P, Q);
		if let Err(s) = T.check() {
			panic!("TableauPair: {}", s);
		}
		T
	}

	/// P in (P, Q)
	pub fn value_tableau(&self) -> Tableau {
		self.0.clone()
	}

	/// Q in (P, Q)
	pub fn index_tableau(&self) -> Tableau {
		self.1.clone()
	}
	pub fn insertion_tableau(&self) -> Tableau {self.index_tableau()}

	pub fn to_two_rowed_array(&self) -> TwoRowedArray {
		let mut v = Vec::new();

		#[allow(non_snake_case)]
		let mut P = self.value_tableau();
		#[allow(non_snake_case)]
		let mut Q = self.index_tableau();

		while let Some(row_index) = Q.greatest_row() {
			// println!("{}", row_index);
			v.push((Q.pop_at_row(row_index), P.reverse_bumping(row_index)));
		}

		v.reverse();
		TwoRowedArray::from_pairs(v)
	}

	// interchange the P and Q in (P, Q),
	pub fn rev(&self) -> TableauPair {
		TableauPair::from(self.index_tableau(), self.value_tableau())
	}
}
impl TwoRowedArray {
	pub fn to_tableau_pair(&self) -> TableauPair {
		#[allow(non_snake_case)]
		let mut P = Tableau::new();
		#[allow(non_snake_case)]
		let mut Q = Tableau::new();
		for (index, value) in self.lexicographic_ordered().iter() {
			Q.place_at_row(P.row_bumping(value), index);
		}
		
		TableauPair(P, Q)
	}
}
#[test] fn conversion() {
	let array = TwoRowedArray::from_two_arrays(vec![1,1,1,2,2,3,3,3,3], vec![1,2,2,1,2,1,1,1,2]).lexicographic_ordered();
	let t = array.to_tableau_pair();
	assert_eq!(t.to_two_rowed_array(), array);

	assert_eq!(t.rev(), array.inverse().to_tableau_pair());
}

mod matrix;
pub use matrix::{Matrix, BallMatrix};

impl TwoRowedArray {
	pub fn to_matrix_0(&self) -> Matrix {
		if self.is_empty() {
			Matrix::new()
		} else {
			let m = *self.index_row().iter().max().unwrap() + 1;
			let n = *self.value_row().iter().max().unwrap() + 1;
	
			let mut matrix = Matrix::from_layout(m, n);

			for (i, j) in self.iter() {
				matrix[(i, j)] += 1;
			}

			matrix
		}

	}
	pub fn to_matrix_1(&self) -> Matrix {
		if self.is_empty() {
			Matrix::new()
		} else {
			let m = *self.index_row().iter().max().unwrap();
			let n = *self.value_row().iter().max().unwrap();
	
			let mut matrix = Matrix::from_layout(m, n);

			for (i, j) in self.iter() {
				matrix[(i - 1, j - 1)] += 1;
			}

			matrix
		}

	}
}
impl Matrix {
	pub fn to_two_rowed_array_0(&self) -> TwoRowedArray {
		let mut array = TwoRowedArray::new();
		for i in 0..self.height() { // rows
			for j in 0..self.width() { // cols
				for _ in 0..self[(i, j)] {
					array.push((i, j));
					println!("{},{}:{}", i, j, array);
				}
			}
		}
		array
	}

	pub fn to_two_rowed_array_1(&self) -> TwoRowedArray {
		let mut array = TwoRowedArray::new();
		for i in 0..self.height() { // rows
			for j in 0..self.width() { // cols
				for _ in 0..self[(i, j)] {
					array.push((i + 1, j + 1));
				}
			}
		}
		array
	}
}
#[test] fn matrix_and_array() {
	let array = TwoRowedArray::from_two_arrays(vec![1,1,1,2,2,3,3,3,3], vec![1,2,2,1,2,1,1,1,2]);
	println!("{}", array);
	println!("{}", array.to_matrix_1());
	assert_eq!(array, array.to_matrix_0().to_two_rowed_array_0());
	assert_eq!(array, array.to_matrix_1().to_two_rowed_array_1());
}

impl BallMatrix {
	/// get the a row for P in (P, Q)
	fn read_col_0(&self) -> Vec<usize> {
		let mut v = Vec::new();
		
		for i in self.number_range() { // the largest markup on the balls
			for col in 0..self.width() {

				if i < self[(self.height() - 1, col)].end {
					// if i is smaller than (or equal to) the bigest element is this col, it must appear in this col or past cols
					v.push(col);
					break;
				}
				// if i is bigger than the bigest element is this col (>= the b in [a, b) of bottom block), go to next col
			}
		}
		v
	}

	/// get the a row for P in (P, Q)
	fn read_col_1(&self) -> Vec<usize> {
		let mut v = Vec::new();
		
		for i in self.number_range() { // the largest markup on the balls
			for col in 0..self.width() {

				if i < self[(self.height() - 1, col)].end {
					// if i is smaller than (or equal to) the bigest element is this col, it must appear in this col or past cols
					v.push(col + 1);
					break;
				}
				// if i is bigger than the bigest element is this col (>= the b in [a, b) of bottom block), go to next col
			}
		}
		v
	}
	/// get the a row for Q in (P, Q)
	fn read_row_0(&self) -> Vec<usize> {
		let mut v = Vec::new();
		
		for i in self.number_range() { // the largest markup on the balls
			for row in 0..self.height() {

				if i < self[(row, self.width() - 1)].end {
					// if i is smaller than (or equal to) the bigest element is this row, it must appear in this row or past rows
					v.push(row);
					break;
				}
				// if i is bigger than the bigest element is this row (>= the b in [a, b) of bottom block), go to next row
			}
		}
		v
	}

	/// get the a row for Q in (P, Q)
	fn read_row_1(&self) -> Vec<usize> {
		let mut v = Vec::new();
		
		for i in self.number_range() { // the largest markup on the balls
			for row in 0..self.height() {

				if i < self[(row, self.width() - 1)].end {
					// if i is smaller than (or equal to) the bigest element is this row, it must appear in this row or past rows
					v.push(row + 1);
					break;
				}
				// if i is bigger than the bigest element is this row (>= the b in [a, b) of bottom block), go to next row
			}
		}
		v
	}
}

impl Matrix {
	pub fn to_tableau_pair_0(&self) -> TableauPair {
		let mut vec_value = Vec::new();
		let mut vec_index = Vec::new();
		let mut bm = self.to_ball_matrix_0();
		while !bm.is_empty() {
			vec_value.push(bm.read_col_0());
			vec_index.push(bm.read_row_0());
			bm = bm.to_new_ball_matrix_0();
		}
		
		TableauPair::from(Tableau::from(vec_value), Tableau::from(vec_index))
	}

	pub fn to_tableau_pair_1(&self) -> TableauPair {
		let mut vec_value = Vec::new();
		let mut vec_index = Vec::new();
		let mut bm = self.to_ball_matrix_1();
		while !bm.is_empty() {
			vec_value.push(bm.read_col_1());
			vec_index.push(bm.read_row_1());
			bm = bm.to_new_ball_matrix_1();
		}
		
		TableauPair::from(Tableau::from(vec_value), Tableau::from(vec_index))
	}
}

#[test] fn matrix_and_tableau_pair() {
	let m = Matrix::from(vec![1, 2, 1, 1, 3, 1], 3, 2);
	// println!("{:?}", m.to_ball_matrix_1().read_col_1());
	// println!("{:?}", m.to_ball_matrix_0().read_col_1());
	assert_eq!(m.to_ball_matrix_1().read_col_1(), m.to_ball_matrix_0().read_col_1());
	assert_eq!(m.to_tableau_pair_1().rev(), m.transpose().to_tableau_pair_1());

	let array = TwoRowedArray::from_two_arrays(vec![1,1,1,2,2,3,3,3,3], vec![1,2,2,1,2,1,1,1,2]);
	// println!("{}", array.to_tableau_pair());
	assert_eq!(array.to_tableau_pair(), array.to_matrix_0().to_tableau_pair_0());
	assert_eq!(array.to_tableau_pair(), array.to_matrix_1().to_tableau_pair_1());
}

impl TableauPair {
	pub fn to_matrix_0() -> Matrix {
		unimplemented!()
	}
}
// */