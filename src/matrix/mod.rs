use std::fmt;
use std::cmp::{max, Ordering};
use std::ops::{Index, IndexMut};

use super::{MathClass};


// ----------------------------------------------------------------
///   --- n ---
/// |  ■ ■ ■ ■
/// m  ■ ■ ■ ■
/// |  ■ ■ ■ ■
#[derive(PartialEq)]
struct Layout {
	pub m : usize,
	pub n : usize,
}
impl fmt::Debug for Layout {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{} × {}", self.m, self.n)
	}
}
impl fmt::Display for Layout {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}",
			format!("{}\n", "■ ".repeat(self.n)).as_str().repeat(self.m)
		)
    }
}
#[test] fn display_layout() {
	let l = Layout {m : 3, n : 2};
	assert_eq!(format!("{}", l), "■ ■ \n■ ■ \n■ ■ \n");
}
impl Layout {
	pub fn index_of(&self, row : usize, col : usize) -> usize {
		col + row * self.n
	}

	pub fn col_of(&self, index : usize) -> usize {
		index % self.n
	}
	pub fn row_of(&self, index : usize) -> usize {
		index / self.n
	}

	pub fn west_one(&self, index : usize) -> Option<usize> {
		if index % self.n == 0 {
			None
		} else {
			Some(index - 1)
		}
	}
	#[allow(dead_code)]
	pub fn east_one(&self, index : usize) -> Option<usize> {
		if index % self.n == self.n - 1 {
			None
		} else {
			Some(index + 1)
		}
	}
	pub fn north_one(&self, index : usize) -> Option<usize> {
		if index < self.n {
			None
		} else {
			Some(index - self.n)
		}
	}
	pub fn south_one(&self, index : usize) -> Option<usize> {
		if index >= self.m * self.n - self.n {
			None
		} else {
			Some(index + self.n)
		}
	}

	pub fn southwest_one(&self, index : usize) -> Option<usize> {
		if let Some(south_one) = self.south_one(index) {
			self.west_one(south_one)
		} else {
			None
		}
	}

	pub fn transpose(&self) -> Layout {
		Layout {m : self.n, n : self.m}
	}
	
	pub fn transpose_index(&self, index : usize) -> usize {
		self.transpose().index_of(self.col_of(index), self.row_of(index))
	}
}
#[test] fn coordinate() {
	let layout = Layout {m : 3, n : 2};
	// println!("{}", layout);
	assert_eq!(layout.index_of(0, 0), 0);
	assert_eq!(layout.index_of(0, 1), 1);
	assert_eq!(layout.index_of(2, 1), 5);
	for index in 0..(layout.m * layout.n) {
		assert_eq!(layout.index_of(layout.row_of(index), layout.col_of(index)), index);

		assert_eq!(layout.transpose().transpose_index(layout.transpose_index(index)), index);
	}

	assert_eq!(None, layout.south_one(4));
	assert_eq!(None, layout.south_one(5));
	assert_eq!(None, layout.north_one(0));
	assert_eq!(None, layout.north_one(1));
	assert_eq!(None, layout.west_one(0));
	assert_eq!(None, layout.west_one(2));
	assert_eq!(None, layout.west_one(4));
	assert_eq!(None, layout.east_one(1));
	assert_eq!(None, layout.east_one(3));
	assert_eq!(None, layout.east_one(5));
}

///   --- n ---
/// |  1 2 3 4
/// m  3 4 3 2
/// |  2 3 4 5
#[derive(Debug, PartialEq)]
pub struct Matrix {
	inner : Vec<usize>,
	layout : Layout,
}
impl MathClass for Matrix {
	fn check(&self) -> Result<(), String> {
		if self.inner.len() == self.layout.m * self.layout.n {
			Ok(())
		} else {
			Err("Matrix: size of matrix differs from the layout".into())
		}
	}
}
impl Matrix {
	pub fn new() -> Matrix {
		Matrix {
			inner : Vec::new(),
			layout : Layout {n : 0, m : 0}
		}
	}

	pub fn from(v : Vec<usize>, m : usize, n : usize) -> Matrix {
		let matrix = Matrix {
			inner : v,
			layout : Layout {n, m},
		};
		if let Err(s) = matrix.check() {
			panic!(s);
		}
		matrix
	}

	pub fn from_layout(m : usize, n : usize) -> Matrix {
		Matrix {
			inner : vec![0 ; m * n],
			layout : Layout {n, m},
		}
	}

	pub fn transpose(&self) -> Matrix {
		let length = self.layout.m * self.layout.n;
		let mut v = vec![0 ; length];
		for index in 0..length {
			v[self.layout.transpose_index(index)] = self.inner[index];
		}
		Matrix::from(v, self.layout.n, self.layout.m)
	}

	fn is_empty(&self) -> bool {
		self.inner.is_empty()
	}

	pub fn is_zero(&self) -> bool {
		self.is_empty() || self.inner.iter().all(|e| *e == 0)
	}

	pub fn width(&self) -> usize {
		self.layout.n
	}
	pub fn height(&self) -> usize {
		self.layout.m
	}
}
impl fmt::Display for Matrix {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let iter = self.inner.chunks(self.layout.n);
		write!(f, "┌ {}┐\n{}└ {}┘",
			"  ".repeat(self.layout.n),

			iter.fold(String::new(), |acc, line| {
				format!("{}{}│\n", acc, line.iter().fold(String::from("│ "), |acc, item| {format!("{}{} ", acc, item)}))
			}),
			
			"  ".repeat(self.layout.n)
		)

		// write!(f, "┌ {}┐\n","  ".repeat(self.layout.n))?;
		// write!(f, "{}",
		// 	format!("│ {}│\n", "■ ".repeat(self.layout.n)).as_str().repeat(self.layout.m)
		// )?;
		// write!(f, "└ {}┘","  ".repeat(self.layout.n))
	}
}
impl Index<(usize, usize)> for Matrix {
	type Output = usize;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.inner[self.layout.index_of(index.0, index.1)]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
	fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
		&mut self.inner[self.layout.index_of(index.0, index.1)]
	}
}

type Range = std::ops::Range<usize>;

///        --- n ---
///    1   | 3   | 4
/// |    2 |     |   5
/// m  --- | --- | ---
/// |  3   | 5   | 6
///      4 |     |   7
#[derive(Debug, PartialEq)]
pub struct BallMatrix {
	inner : Vec<Range>,
	layout : Layout,
}
impl MathClass for BallMatrix {
	fn check(&self) -> Result<(), String> {
		if self.inner.len() != self.layout.m * self.layout.n {
			Err("BallMatrix: size of matrix differs from the layout".into())
		} else {
			for index in 0..self.inner.len() {
				if !match (self.layout.west_one(index), self.layout.north_one(index)) {
					(None, None) => true,
					(Some(west_one), None) => self.inner[index].start == self.inner[west_one].end,
					(None, Some(north_one)) => self.inner[index].start == self.inner[north_one].end,
					(Some(west_one), Some(north_one)) => self.inner[index].start == max(self.inner[west_one].end, self.inner[north_one].end),
				} {
					return Err("BallMatrix: error on the markup of Balls ".into())
				}
			}
			Ok(())
		}
	}
}
impl BallMatrix {
	pub fn from(v : Vec<(usize, usize)>, m : usize, n : usize) -> BallMatrix {
		let matrix = BallMatrix {
			inner : v.into_iter().map(|(a, b)| {a..b}).collect(),
			layout : Layout {n, m},
		};
		if let Err(s) = matrix.check() {
			panic!(s);
		}
		matrix
	}
	pub fn is_empty(&self) -> bool {
		self.inner.is_empty() || self.inner.last().unwrap().end - self.inner[0].start == 0 // ?
	}

	pub fn width(&self) -> usize {
		self.layout.n
	}
	pub fn height(&self) -> usize {
		self.layout.m
	}

	pub fn number_range(&self) -> Range {
		if self.is_empty() {
			0..0
		} else {
			self.inner[0].start..self.inner.last().unwrap().end
		}
	}
}

impl Index<(usize, usize)> for BallMatrix {
	type Output = Range;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.inner[self.layout.index_of(index.0, index.1)]
    }
}
impl IndexMut<(usize, usize)> for BallMatrix {
	fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
		&mut self.inner[self.layout.index_of(index.0, index.1)]
	}
}


impl fmt::Display for BallMatrix {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let iter = self.inner.chunks(self.layout.n);
		write!(f, "{}",
			iter.fold(String::new(), |acc, line| {
				format!("{}{}\n {}\n", acc, line.iter().fold(String::new(), |acc, item| {format!("{}{:#?}\t|", acc, item)}), "-------|".repeat(self.layout.n))
			})
		)
	}
}
#[test] fn ball_matrix_from_tuple() {
	assert!(
		std::panic::catch_unwind(|| {
			BallMatrix::from(vec![
				(1,2), (1,2),
				(2,3), (3,4),
				(3,5), (5,6),
			], 3, 2);
		}).is_err()
	);

	// println!("{}", BallMatrix::from(vec![ (1,2), (2,2), (2,3), (3,4), (3,5), (5,6), ], 3, 2));
}

fn compare(range : &Range, k : usize) -> Ordering {
	if k < range.start {
		Ordering::Less
	} else if k >= range.end {
		Ordering::Greater
	} else {
		Ordering::Equal
	}
}

impl BallMatrix {
	pub fn to_matrix(&self) -> Matrix {
		Matrix::from(self.inner.iter().map(|e| {e.len()}).collect(), self.layout.m, self.layout.n)
	}

	pub fn to_new_matrix(&self) -> Matrix {
		let mut matrix = Matrix::from_layout(self.layout.m, self.layout.n);
		// * I run it from above to below rather than left to right in the book page 43
		for index in 0..self.inner.len() {
			
			for k in self.inner[index].clone() { // ? clone // the markup of ball (a ball numbered with k)
				if let Some(mut another) = self.layout.southwest_one(index) {
					loop {
						another = match compare(&self.inner[another], k) {
							Ordering::Less => if let Some(west_one) = self.layout.west_one(another) {
								west_one
							} else {
								break;
							},
							Ordering::Greater => if let Some(south_one) = self.layout.south_one(another) {
								south_one
							} else {
								break;
							},
							Ordering::Equal => {
								matrix[(self.layout.row_of(another), self.layout.col_of(index))] += 1;
								// simply speaking, there will be a new ball in the the block in the col of index block and row of another block
								break;
							}
						}
					}
				}
			}
		}

		matrix
	}

	pub fn to_new_ball_matrix_0(&self) -> BallMatrix {
		self.to_new_matrix().to_ball_matrix_0()
	}

	pub fn to_new_ball_matrix_1(&self) -> BallMatrix {
		self.to_new_matrix().to_ball_matrix_1()
	}
}
#[test] fn matrix_and_ball_matrix() {
	let m = Matrix::from(vec![1, 2, 1, 1, 3, 1], 3, 2);
	// println!("{}", m.to_ball_matrix_1());
	assert_eq!(m.to_ball_matrix_1().to_matrix(), m);
	assert_eq!(m.to_ball_matrix_0().to_matrix(), m);
	assert_ne!(m.to_ball_matrix_1(), m.to_ball_matrix_0());

	// println!("{}", m);
	// println!("{}", m.to_new_matrix());
	// println!("{}", m.to_new_matrix().to_new_matrix());
	assert!(m.to_new_matrix().to_new_matrix().is_zero())
}
impl Matrix {
	pub fn to_ball_matrix_0(&self) -> BallMatrix {
		let mut v = vec![(0,0) ; self.inner.len()];

		for index in 0..self.inner.len() {
			v[index].0 = match (self.layout.west_one(index), self.layout.north_one(index)) {
				(None, None) => 0, // *************** to_ball_matrix_0
				(None, Some(north_one)) => v[north_one].1,
				(Some(west_one), None) => v[west_one].1,
				(Some(west_one), Some(north_one)) => max(v[north_one].1, v[west_one].1),
			};
			v[index].1 = v[index].0 + self.inner[index];
		}

		BallMatrix::from(v, self.layout.m, self.layout.n)
	}

	pub fn to_ball_matrix_1(&self) -> BallMatrix {
		let mut v = vec![(0,0) ; self.inner.len()];

		for index in 0..self.inner.len() {
			v[index].0 = match (self.layout.west_one(index), self.layout.north_one(index)) {
				(None, None) => 1, // *************** to_ball_matrix_1
				(None, Some(north_one)) => v[north_one].1,
				(Some(west_one), None) => v[west_one].1,
				(Some(west_one), Some(north_one)) => max(v[north_one].1, v[west_one].1),
			};
			v[index].1 = v[index].0 + self.inner[index];
		}

		BallMatrix::from(v, self.layout.m, self.layout.n)
	}

	pub fn to_new_matrix(&self) -> Matrix {
		self.to_ball_matrix_0().to_new_matrix()
		// which is equal to
		// self.to_ball_matrix_1().to_new_matrix()
	}
}

