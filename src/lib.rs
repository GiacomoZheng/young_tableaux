// + SkewTableau

use std::fmt;
use std::cmp::{PartialEq, PartialOrd};
use std::ops::{Mul};

mod tools;
use tools::order::{is_strictly_increasing, is_weakly_increasing, replace_greatest_predecessor, replace_least_successor};
use tools::VecTail;
use tools::MathClass;

#[derive(PartialEq, PartialOrd, Eq)]
pub struct Diagram(VecTail<usize>);
impl MathClass for Diagram {
	fn check(&self) -> Result<(), String> {
		if let Err(s) = self.0.is_weakly_decreasing() {
			Err(format!("young diagram {}", s))
		} else {
			Ok(())
		}
	}
}
impl Diagram {
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}
	
	pub fn from(v : Vec<usize>) -> Diagram {
		let diag = Diagram(VecTail::from(v, 0usize));
		if let Err(s) = diag.check() {
			panic!("Diagram: {}", s);
		}
		diag
	}

	/// n(6, 4, 4, 2) = 6 + 4 + 4 + 2 = 16
	pub fn n(&self) -> usize {
		self.0.len()
	}

	/// |(6, 4, 4, 2)| = 4
	pub fn abs(&self) -> usize {
		self.0.iter_finite().count()
	}

	pub fn transpose(&self) -> Diagram {
		let mut list = VecTail::new();
		// println!("{:?}", self);
		// println!("{}", self);
		for i in 0..*self.0.iter().next().unwrap()  {
			let depth = self.0.iter_finite().take_while(|e| {**e > i}).count();
			list.push(depth);
		}
		Diagram(list)
	}

	/// ? an increasing vec
	pub fn rows_of_corners(&self) -> Vec<usize> {
		// self.0.iter().enumerate().tuple_windows::<(_,_)>().filter(|((_, prev), (_, next))| {prev > next}).take_while(|((_, e), _)| {**e > 0}).inspect(|x| {println!("{:?}: e > 0", x)}).map(|((index, _), _)| {index}).collect::<Vec<usize>>()
		let mut rows = Vec::new();
		let mut iter = self.0.iter().enumerate().peekable();
		let mut this = iter.next();
		let mut next = iter.peek();
		loop {
			if *this.unwrap().1 > *next.unwrap().1 {
				rows.push(this.unwrap().0);
			} else if *this.unwrap().1 == 0 {
				break;
			}
			this = iter.next();
			next = iter.peek();
		}
		rows
	}
	pub fn rows_below_corners(&self) -> Vec<usize> {
		self.rows_of_corners().into_iter().map(|x| {x + 1}).collect()
	}
}
#[test]
fn transpose() {
	let diagram = Diagram::from(vec![6,6,3,2,1]);
	assert_eq!(diagram.transpose(), Diagram::from(vec![5,4,3,2, 2,2]));
}
#[test]
fn corners() {
	let diagram = Diagram::from(vec![3,2,2,1]);
	assert_eq!(diagram.rows_of_corners(), vec![0,2,3]);
}

impl fmt::Debug for Diagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl fmt::Display for Diagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for &len in self.0.iter().take_while(|len| {**len > 0}) {
			writeln!(f, "{}", "■ ".repeat(len))?;
		}
		write!(f, "")
    }
}
#[test]
fn display_diagram() {
	let diagram = Diagram::from(vec![3,2,1]);
	// println!("{}", diagram);
	assert_eq!(format!("{}", diagram), String::from("■ ■ ■ \n■ ■ \n■ \n"));
}
#[test]
fn order() {
	let diagram = Diagram::from(vec![5,4,1]);
	let diagram_e = Diagram::from(vec![5,4,1]);
	let diagram_l = Diagram::from(vec![4,4,1]);

	// println!("{}", diagram);
	// println!("{}", diagram_l);
	assert!(diagram_e == diagram);
	assert!(diagram_l <= diagram);
}

#[derive(PartialEq)]
pub struct SkewDiagram {
	inner : Diagram,
	outer : Diagram,
}
impl MathClass for SkewDiagram {
	fn check(&self) -> Result<(), String> {
		for (&i, &o) in self.inner.0.iter().take_while(|i| {**i > 0}).zip(self.outer.0.iter()) {
			if i > o {
				return Err("the inner Diagram should be \"inside\" of outer Diagram".into())
			}
		}
		Ok(())
	}
}
impl SkewDiagram {
	pub fn from(inner : Diagram, outer : Diagram) -> SkewDiagram {
		let skew_diagram = SkewDiagram {
			inner,
			outer,
		};
		
		if let Err(s) = skew_diagram.check(){
			panic!("SkewDiagram: {}", s);
		}
		skew_diagram
	}

	/// the common corners of inner and outer
	pub fn common_corners(&self) -> Vec<usize> {
		let inner_corners = self.inner.rows_of_corners();
		let outer_corners = self.outer.rows_of_corners();
		self.inner.0.iter().take_while(|i| {**i > 0}).zip(self.outer.0.iter()).enumerate().filter(|(_, (&i, &o))| {i == o}).map(|(row, _)| {row}).filter(|row| {inner_corners.contains(row) && outer_corners.contains(row)}).collect()
	}
}
impl fmt::Debug for SkewDiagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "λ: {}, μ: {}", format!("{:?}", self.inner), format!("{:?}", self.outer))
    }
}
impl fmt::Display for SkewDiagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for (&inner_bound, &outer_bound) in self.inner.0.iter().zip(self.outer.0.iter()).take_while(|(_, &e)| e > 0) {
			writeln!(f, "{}{}", "□ ".repeat(inner_bound), "■ ".repeat(outer_bound - inner_bound))?;
		}
		write!(f, "")
    }
}
#[test]
fn print_skew_diagram() {
	let skew_diagram = SkewDiagram::from(
		Diagram(VecTail::from(vec![2,1], 0)),
		Diagram(VecTail::from(vec![3,2,1], 0)),
	);
	// println!("{}", skew_diagram);
	assert_eq!(format!("{}", skew_diagram), String::from("□ □ ■ \n□ ■ \n■ \n"));
	assert_eq!(format!("{:?}", skew_diagram), String::from("λ: (2, 1, 0...), μ: (3, 2, 1, 0...)"));
}

// -------------------------------------------------------------
#[derive(PartialEq, Debug, Clone)]
pub struct SkewTableau(VecTail<Vec<Option<usize>>>);
impl MathClass for SkewTableau {
	// ? to complicated
	fn check(&self) -> Result<(), String> {
		self.shape().check()?;

		for row in self.0.iter().take_while(|e| {**e != Vec::new()}) {
			if let Err(s) = is_weakly_increasing(row) {
				return Err(format!("rows in tableau {}", s));
			}
		}

		for col in self.pre_transpose().iter().take_while(|e| {**e != Vec::new()}) {
			if let Err(s) = is_strictly_increasing(col) {
				return Err(format!("cols in tableau {}", s))
			}
		}
		Ok(())
	}
}
impl SkewTableau {
	pub fn new() -> SkewTableau {
		SkewTableau(VecTail::new())
	}

	pub fn from(v : Vec<Vec<Option<usize>>>) -> SkewTableau {
		let tableau = SkewTableau(VecTail::from(v, Vec::new()));
		
		if let Err(s) = tableau.check() {
			panic!("SkewTableau: {}", s);
		}
		tableau
	}

	fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	// ? to complicated
	fn pre_transpose(&self) -> VecTail<Vec<Option<usize>>> {
		let mut v : Vec<Vec<Option<usize>>> = Vec::new();
		for index in 0..self.0.iter().next().unwrap().len() {
			let mut tmp_v = Vec::new();
			for row in self.0.iter() {
				if index < row.len() {
					if let Some(e) = row[index] {
						tmp_v.push(Some(e));
					}
				} else {
					break;
				}
			}
			v.push(tmp_v);
		}
		VecTail::from(v, Vec::new())
	}

	pub fn shape(&self) -> SkewDiagram {
		SkewDiagram::from(
			Diagram::from(self.0.iter_finite().map(|v| {v.iter().take_while(|e| {e.is_none()}).count()}).collect()),
			Diagram::from(self.0.iter_finite().map(|v| {v.iter().count()}).collect()),
		)
	}
}

#[test]
fn check_skew_tableau() {
	assert!(
		std::panic::catch_unwind(|| {
			SkewTableau::from(vec![
				vec![   None, Some(1),    None, Some(2)],
				vec![   None, Some(2)],
				vec![Some(2)],
			]);
		}).is_err()
	)
}
#[test]
fn shape() {
	let tableau = SkewTableau::from(vec![
        vec![   None,    None, Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	
	// println!("{}", tableau.shape());
	assert_eq!(tableau.shape(), SkewDiagram {
		inner : Diagram(VecTail::from(vec![2], 0)),
		outer : Diagram(VecTail::from(vec![3,2,1], 0)),
	});
}

impl SkewTableau {
	/// or `row_insert`
	/// return the row_index of the final process
	pub fn row_bumping(&mut self, x : usize) -> usize {
		let mut bumped = x; // it would become bigger and bigger 
		let mut row_index = 0;
		while let Some(tmp) = replace_least_successor(bumped, &mut self.0[row_index]) {
			bumped = tmp;
			row_index += 1;
		}
		self.0[row_index].push(Some(bumped));
		row_index
	}
		pub fn row_insert(&mut self, x : usize) -> usize {
			self.row_bumping(x)
		}

	/// ? to complicated
	/// return the value inserted 
	pub fn reverse_bumping(&mut self, row_index : usize) -> usize {
		if let Some(Some(mut bumped)) = self.0[row_index].pop() {
			for row in self.0.iter_finite_mut().take(row_index).rev() { // ?
				if let Some(tmp) = replace_greatest_predecessor(bumped, row) {
					bumped = tmp;
				} else {
					panic!("it is not a well defined tableau")
				}
			}
			bumped
		} else {
			panic!("no possible \"new block\" in this row")
		}
	}

	/// return the index of the removed corner
	pub fn sliding(&mut self, mut row_index : usize) -> usize {
		// println!("{}", self);
		if self.shape().inner.rows_of_corners().contains(&row_index) {
			let mut col_index = self.shape().inner.0[row_index] - 1; // the last one
			// `hole` is self.0[row_index][col_index]
			loop {
				// println!("{}", self);
				// println!("hole: ({}, {})", col_index, row_index);
				let below_one = {
					let next_row = &mut self.0[row_index + 1];
					if let Some(e) = next_row.get_mut(col_index) {
						e.take()
					} else {
						None
					}
				};
				// println!("below one: {:?}", below_one);
				let right_one = match self.0[row_index].get_mut(col_index + 1) {
					Some(e) => e.take(),
					None => None
				};
				// println!("right one: {:?}", right_one);
				
				if (below_one, right_one) == (None, None) {
					self.0[row_index].pop(); // remove the hole
					// println!("end");
					break row_index;

				} else if (right_one.is_some() && right_one < below_one) || below_one.is_none() {
					self.0[row_index][col_index] = right_one;
					if below_one.is_some() {
						self.0[row_index + 1][col_index] = below_one;
					}
					col_index += 1;
					// println!("move to right");
				} else if (below_one.is_some() && right_one >= below_one) || right_one.is_none(){
					// if they are equal, move to the below one
					self.0[row_index][col_index] = below_one;
					if right_one.is_some() {
						self.0[row_index][col_index + 1] = right_one;
					}
					row_index += 1;
					// println!("move to below");
				} else {
					panic!("Tell me, WTH is the else case?!")
				}
				
			}
		} else {
			panic!("no inner corners in this row")
		}
	}

	/// ! unstable: `self.0.len()`
	/// Rect(S)
	/// jeu de taquin
	/// a rectification (redressement) of a skew tableau S
	pub fn rect(&mut self) {
		// do until self.shape().inner is empty
		// I'll repeat on last non-empty inner line, which is always a corner
		let list = self.shape().inner.0;
		for row_index in (0..list.len()).rev() { // ?
			for _ in 0..list[row_index] {
				// println!("start a new hole at {}", row_index);
				self.sliding(row_index);
			}

		}
	}

	/// note: `reverse` of sliding(1) may not be the reverse_sliding(1), it should be reverse_sliding(sliding(1))
	/// return the row of new box
	pub fn reverse_sliding(&mut self, mut row_index : usize) -> usize { // +
		// println!("{}", self);
		if self.shape().outer.rows_below_corners().contains(&row_index) || row_index == 0 {
			let mut col_index = *self.shape().outer.0.iter().nth(row_index).unwrap(); // the one right to the last one
			// `hole` is self.0[row_index][col_index]
			self.0[row_index].push(None);
			loop {
				// println!("{}", self);
				// println!("hole: ({}, {})", col_index, row_index);
				let above_one = if row_index == 0 {
					None
				} else {
					self.0[row_index - 1][col_index].take()
				};
				// println!("above one: {:?}", above_one);
				#[allow(non_snake_case)]
				let left__one = if col_index == 0 {
					None
				} else {
					self.0[row_index][col_index - 1].take()
				};
				// println!("left one: {:?}", left__one);
				
				if (above_one, left__one) == (None, None) {
					// println!("end");
					break row_index;

				} else if (left__one.is_some() && left__one > above_one) || above_one.is_none() {
					self.0[row_index][col_index] = left__one;
					if above_one.is_some() {
						self.0[row_index - 1][col_index] = above_one;
					}
					col_index -= 1;
					// println!("move to left");
				} else if (above_one.is_some() && left__one <= above_one) || left__one.is_none(){
					// if they are equal, move to the below one
					self.0[row_index][col_index] = above_one;
					if left__one.is_some() {
						self.0[row_index][col_index - 1] = left__one;
					}
					row_index -= 1;
					// println!("move to above");
				} else {
					panic!("Tell me, WTH is the else case?!")
				}
				
			}
		} else {
			panic!("no potential common_corners in this row")
		}
	}
}
#[test]
fn bumping() {
	let mut tableau = SkewTableau::from(vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	// println!("{}", tableau);
	
	let r1 = tableau.row_bumping(2);
	// println!("{}", tableau);
	assert_eq!(tableau, SkewTableau::from(vec![
        vec![Some(1), Some(2), Some(2)],
        vec![Some(2), Some(3), Some(3)],
        vec![Some(4)],
	]));

	let r2 = tableau.row_bumping(1);
	// println!("{}", tableau);
	assert_eq!(tableau, SkewTableau::from(vec![
        vec![Some(1), Some(1), Some(2)],
        vec![Some(2), Some(2), Some(3)],
        vec![Some(3)],
        vec![Some(4)],
	]));
	
	assert_eq!(tableau.reverse_bumping(r2), 1);
	// println!("{}", tableau);
	assert_eq!(tableau, SkewTableau::from(vec![
        vec![Some(1), Some(2), Some(2)],
        vec![Some(2), Some(3), Some(3)],
        vec![Some(4)],
	]));
	
	assert_eq!(tableau.reverse_bumping(r1), 2);
	// println!("{}", tableau);
	assert_eq!(tableau, SkewTableau::from(vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]));
}
#[test]
fn sliding() {
	let mut skew_tableau = SkewTableau::from(vec![
        vec![   None,    None, Some(2)],
        vec![   None, Some(6)],
        vec![Some(5)],
	]);
	// println!("{}", skew_tableau);
	// go right
	let r1 = skew_tableau.sliding(0);
	assert_eq!(skew_tableau, SkewTableau::from(vec![
        vec![   None, Some(2)],
        vec![   None, Some(6)],
        vec![Some(5)],
	]));

	// go below
	let r2 = skew_tableau.sliding(1);
	assert_eq!(skew_tableau, SkewTableau::from(vec![
        vec![   None, Some(2)],
        vec![Some(5), Some(6)],
	]));

	assert_eq!(skew_tableau.reverse_sliding(r2), 1);
	assert_eq!(skew_tableau.reverse_sliding(r1), 0);
	assert_eq!(skew_tableau, SkewTableau::from(vec![
        vec![   None,    None, Some(2)],
        vec![   None, Some(6)],
        vec![Some(5)],
	]));

	let mut skew_tableau = Word(vec![1,2,42,4,5,3,3,4,3,4,2,233,2]).to_skew_tableau();
	// println!("{}", skew_tableau);

	skew_tableau.rect();
	// println!("{}", tableau);
	let tableau = skew_tableau;
	assert_eq!(tableau,SkewTableau::from(vec![
		vec![Some(1), Some(2), Some(2), Some(2), Some(3), Some(4), Some(233)],
		vec![Some(3), Some(3)],
		vec![Some(4), Some(4)],
		vec![Some(5)],
		vec![Some(42)],
	]));
}

impl Mul for SkewTableau {
	type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
		for &r in rhs.to_word().0.iter() {
			self.row_bumping(r);
		}
		self
    }
}
#[test]
fn mul_tableau() {
	let lhs = Word(vec![1,2,3,4,3]).to_tableau();
	let rhs = Word(vec![1,2,3,4,3]).to_tableau();
	assert_eq!((lhs * rhs).to_word(), Word(vec![4,2,3,4,1,1,2,3,3,3]));
	
	let lhs = Word(vec![1,2,3,4,3]).to_tableau();
	let rhs = Word(vec![3,6,3,4]).to_tableau();
	assert_eq!((lhs * rhs).to_word(), Word(vec![4,6,1,2,3,3,3,3,4]));
}

impl fmt::Display for SkewTableau {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for v in self.0.iter().take_while(|e| {**e != Vec::new()}) {
			for e in v.iter() {
				if let Some(e) = e {
					write!(f, "{} ", e)?;
				} else {
					write!(f, "□ ")?;
				}
			}
			writeln!(f, "")?;
		}
		write!(f, "")
    }
}
#[test]
fn display_tableau() {
	let tableau = SkewTableau::from(vec![
		vec![   None, Some(6), Some(7), Some(8)],
		vec![Some(5), Some(7), Some(8)],
		vec![Some(6), Some(8)],
	]);
	// println!("{}", tableau);
	assert_eq!(format!("{}", tableau), String::from("□ 6 7 8 \n5 7 8 \n6 8 \n"));
}

#[derive(Debug, PartialEq, Clone)]
pub struct Word(pub Vec<usize>);
impl Word {
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// `L(w, k)` means the largest numberas within the sum of the lengths of `k` disjoint (weakly) increasing sequences extracted from `w`
	#[allow(non_snake_case)]
	pub fn L(&self, k : usize) -> usize {
		self.to_tableau().shape().outer.0.iter().cloned().take(k).sum()
	}
}
#[test]
fn increasing_seq() {
	let word = Word(vec![1, 4, 2, 5, 6, 3]);
	assert_eq!(word.L(0), 0);
	assert_eq!(word.L(1), 4);
	assert_eq!(word.L(2), 6);
	assert_eq!(word.L(3), 6);
	assert_eq!(word.L(4), 6);
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
	pub fn to_tableau(&self) -> SkewTableau {
		let mut tableau = SkewTableau::new();
		for &letter in self.0.iter() {
			tableau.row_insert(letter);
		}
		tableau
	}
}
impl SkewTableau {
	/// w(T) = from left below side to right and row by row upwards
	pub fn to_word(&self) -> Word { // may be -> Word
		Word(self.0.iter_finite().rev().flatten().filter(|e| {e.is_some()}).map(|e| {e.unwrap()}).collect())
	}
}
#[test]
fn word() {
	let tableau = SkewTableau::from(vec![
		vec![   None,Some(4),Some(5),Some(6),Some(7)],
		vec![Some(5),Some(7)],
		vec![Some(7)],
	]);
	
	// println!("{}", tableau);
	let word = tableau.to_word();
	assert_eq!(word, Word(vec![7, 5, 7, 4, 5, 6, 7]));
	
	let skew_tableau = word.to_skew_tableau();
	// println!("{}", skew_tableau);
	assert_eq!(skew_tableau.to_word(), Word(vec![7, 5, 7, 4, 5, 6, 7]));

	assert_eq!(word.to_tableau().to_word(), Word(vec![7, 5, 7, 4, 5, 6, 7]));
}

impl Mul for Word {
	type Output = Self;
    fn mul(mut self, mut rhs: Self) -> Self {
		self.0.append(&mut rhs.0);
		self
    }
}
#[test]
fn mul_equivalence() {
	let lhs = Word(vec![1,2,3,4,2,3,4,9,2,3,2,3,4,2,2]);
	let rhs = Word(vec![1,2,3,42,343,464,334,33,2,3,2,5,3,43,2,1]);
	assert_eq!((lhs.clone() * rhs.clone()).to_tableau(), lhs.to_tableau() * rhs.to_tableau());
}

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
struct Pair<T : PartialEq + Ord>(T, T);
/* impl<T : PartialEq + Ord> PartialOrd for Pair<T> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		if self.0 < other.0 {
			Some(Ordering::Less)
		} else if self.0 > other.0 {
			Some(Ordering::Greater)
		} else {
			if self.1 < other.1 {
				Some(Ordering::Less)
			} else if self.1 > other.1 {
				Some(Ordering::Greater)
			} else {
				Some(Ordering::Equal)
			}
		}
	}
}
impl<T : PartialEq + Ord> Ord for Pair<T> {
	fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
} */
#[test]
fn pair_order() {
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
	pub fn inverse(self) -> TwoRowedArray {
		TwoRowedArray(self.0.into_iter().map(Pair::rev).collect())
	}

	// pub fn push(&mut self, (index, value) : (usize, usize)) {
	// 	self.0.push(Pair::from((index, value)));
	// }
}
impl PartialEq for TwoRowedArray {
	fn eq(&self, other: &Self) -> bool {
        &self.lexicographic_ordered().0 == &other.lexicographic_ordered().0
    }
}
#[test]
fn from_two_arrays() {
	assert_eq!(
		TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]),
		TwoRowedArray::from_pairs(
			vec![(1, 3),(2, 4),(3, 2),(4, 5),(2, 4),(4, 2),(4, 3),(2, 2)],
		)
	);
}
#[test]
fn sort() {
	let array = TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]).lexicographic_ordered();
	assert_eq!(array, TwoRowedArray::from_two_arrays(
		vec![1,2,2,2,3,4,4,4],
		vec![3,2,4,4,2,2,3,5]
	));
}

impl fmt::Display for TwoRowedArray {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}\n{}",
			self.index_row().iter().fold(String::new(), |acc, e| {format!("{}{} ", acc, e)}),
			self.value_row().iter().fold(String::new(), |acc, e| {format!("{}{} ", acc, e)}),
		)
    }
}
#[test]
fn print_array() {
	let array = TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]);
	assert_eq!(format!("{}", array), "1 2 3 4 2 4 4 2 \n3 4 2 5 4 2 3 2 ");
}
#[allow(non_snake_case)]
impl Word {
	pub fn to_TwoRowedArray_0(&self) -> TwoRowedArray {
		TwoRowedArray::from_pairs(self.0.iter().cloned().enumerate().collect())
	}

	pub fn to_TwoRowedArray_1(&self) -> TwoRowedArray {
		TwoRowedArray::from_pairs(self.0.iter().cloned().enumerate().map(|(index, value)| {(index + 1, value)}).collect())
	}
}

// for tableau
impl SkewTableau {
	/// simply put a element on this row
	pub fn place(&mut self, row_index : usize, value : usize) {
		self.0[row_index].push(Some(value));
	}
	
	pub fn pop_at(&mut self, row_index : usize) -> usize {
		if let Some(value) = self.0[row_index].pop() {
			self.0.strip();
			value.unwrap()
		} else {
			panic!("this row is empty, you cannot pop at here");
		}
	}

	/// return the row_index of the greatest element
	/// pick the rightmost within the equals
	/// + for tableau
	pub fn greatest_row(&self) -> Option<usize> {
		// println!("self: {:?}", self);
		// println!("{:?}", self.is_empty());
		if self.is_empty() {
			return None;
		}
		let mut gest : Option<usize> = None;
		let mut aim_index = usize::MAX;
		for index in self.shape().outer.rows_of_corners() {
			let tmp = *self.0.iter().nth(index).unwrap().last().unwrap();
			if tmp > gest {
				aim_index = index;
				// println!("aim: {}", aim_index);
				gest = tmp;
			} else if tmp == gest && index < aim_index {
				aim_index = index;
				// println!("aim: {}", aim_index);
			}
		}
		Some(aim_index)
	}
}
#[test]
fn place_and_pop() {
	let mut tableau = SkewTableau::from(vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	tableau.pop_at(2);
	// println!("{}", tableau);
	tableau.pop_at(1);
	// println!("{}", tableau);
}
#[test]
fn greatest_row() {
	let mut tableau = SkewTableau::from(vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	assert_eq!(tableau.greatest_row(), Some(2));
	tableau.pop_at(2);
	println!("{}", tableau);
	assert_eq!(tableau.greatest_row(), Some(0));
	tableau.pop_at(0);
	println!("{}", tableau);
	assert_eq!(tableau.greatest_row(), Some(1));
}

// + to Tableau
#[derive(Debug)]
pub struct TableauPair(SkewTableau, SkewTableau);
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
	pub fn from(P : SkewTableau, Q : SkewTableau) -> TableauPair {
		let T = TableauPair(P, Q);
		if let Err(s) = T.check() {
			panic!("TableauPair: {}", s);
		}
		T
	}

	/// P in (P, Q)
	pub fn value_tableau(&self) -> SkewTableau {
		self.0.clone()
	}

	/// Q in (P, Q)
	pub fn index_tableau(&self) -> SkewTableau {
		self.1.clone()
	}
	pub fn insertion_tableau(&self) -> SkewTableau {self.index_tableau()}

	#[allow(non_snake_case)]
	pub fn to_TwoRowedArray(&self) -> TwoRowedArray {
		let mut v = Vec::new();

		#[allow(non_snake_case)]
		let mut P = self.value_tableau();
		#[allow(non_snake_case)]
		let mut Q = self.index_tableau();

		while let Some(row_index) = Q.greatest_row() {
			// println!("{}", row_index);
			v.push((Q.pop_at(row_index), P.reverse_bumping(row_index)));
		}

		v.reverse();
		TwoRowedArray::from_pairs(v)
	}
}
impl TwoRowedArray {
	pub fn to_tableau_pair(&self) -> TableauPair {
		#[allow(non_snake_case)]
		let mut P = SkewTableau::new();
		#[allow(non_snake_case)]
		let mut Q = SkewTableau::new();
		for &Pair(index, value) in self.lexicographic_ordered().0.iter() {
			Q.place(P.row_bumping(value), index);
		}
		
		TableauPair(P, Q)
	}
}
#[test]
fn conversion() {
	let array = TwoRowedArray::from_two_arrays(vec![1,1,1,2,2,3,3,3,3], vec![1,2,2,1,2,1,1,1,2]).lexicographic_ordered();
	#[allow(non_snake_case)]
	let T = array.to_tableau_pair();
	assert_eq!(T.to_TwoRowedArray(), array);
}
// */