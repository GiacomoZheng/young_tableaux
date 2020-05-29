use std::fmt;
use std::cmp::{PartialEq, PartialOrd, Ordering};
use std::ops::{Mul, Deref, DerefMut, Index, IndexMut};

#[allow(dead_code)]
mod tools {
	use std::cmp::PartialOrd;
	pub fn check_weakly_decreasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		if !v.is_empty() {
			for index in 0..(v.len() - 1) {
				if v[index + 1] > v[index] {
					return Err("should be weakly deceasing");
				}
			}
		}
		Ok(())
	}
	pub fn check_weakly_increasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		if !v.is_empty() {
			for index in 0..(v.len() - 1) {
				if v[index + 1] < v[index] {
					return Err("should be weakly inceasing");
				}
			}
		}
		Ok(())
	}
	
	pub fn check_strictly_decreasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		if !v.is_empty() {
			for index in 0..(v.len() - 1) {
				if v[index + 1] >= v[index] {
					return Err("should be strictly deceasing");
				}
			}
		}
		Ok(())
	}
	pub fn check_strictly_increasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
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
}
use tools::*;

use std::iter::{repeat};
#[derive(Eq, Clone)]
struct InftyList<T> {
	heads : Vec<T>,
	tail : T,
}
impl<T : Default> InftyList<T> {
	pub fn new() -> InftyList<T> {
		InftyList {
			heads : Vec::new(),
			tail : T::default(),
		}
	}
}
impl<T : PartialEq> InftyList<T> {
	fn check(v : &Vec<T>, tail : &T) {
		if v.contains(tail) {
			panic!("element of tail should not appears in the head")
		}
	}

	pub fn from(v : Vec<T>, tail : T) -> InftyList<T> {
		InftyList::check(&v, &tail);
		InftyList {
			heads : v,
			tail,
		}
	}

	pub fn is_empty(&self) -> bool {
		self.iter().next() == Some(&self.tail)
	}

	pub fn iter(&self) -> std::iter::Chain<std::slice::Iter<'_, T>, std::iter::Repeat<&T>> {
		self.heads.iter().chain(repeat(&self.tail))
	}
	
	/// ! unstale
	fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
		self.heads.iter_mut()
	}

	/// ! unstale
	fn iter_finite<'a>(&self) -> std::slice::Iter<'_, T> {
		self.heads.iter()
	}

	// if push a tail return false
	pub fn push(&mut self, value: T) -> bool {
		if value != self.tail {
			while let Some(t) = self.heads.last() {
				if *t == self.tail {
					self.heads.pop();
				} else {
					break;
				}
			}
			self.heads.push(value);
			true
		} else {
			false
		}
	}
}
impl<T : PartialEq> PartialEq for InftyList<T> {
	fn eq(&self, other: &Self) -> bool {
		self.iter().zip(other.iter()).take_while(|(s, o)| {**s != **o || **o == self.tail}).all(|(s, o)| {*s == *o})
    }
}
impl<T : Clone> Index<usize> for InftyList<T> {
	type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
		if index < self.heads.len() {
			self.heads.index(index)
		} else {
			&self.tail
		}
    }
}
impl<T : Clone> IndexMut<usize> for InftyList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		if index < self.heads.len() {
			self.heads.index_mut(index)
		} else if index == self.heads.len() {
			self.heads.push(self.tail.clone());
			self.heads.last_mut().unwrap()
		} else {
			panic!("You cannot change the tails except the first one")
		}
    }
}
// ? I'm not sure about it
impl<T> Deref for InftyList<T> {
	type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.heads
	}
}
impl<T> DerefMut for InftyList<T> {
	fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.heads
    }
}
impl<T : fmt::Debug + PartialEq> fmt::Debug for InftyList<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "({}{:?}{})", self.iter().take_while(|e| {**e != self.tail}).fold(String::new(), |acc, e| {format!("{}{:?}, ", acc, e)}), self.tail, "...")
	}
}

#[test]
fn print_infty() {
	let list = InftyList::from(vec![1,2,3], 0);
	// println!("{:?}", list);
	assert_eq!(String::from("(1, 2, 3, 0...)"), format!("{:?}", list) );
}
#[test]
fn eq_infty() {
	let inf_1 = InftyList::from(vec![1,2,3,4], 0);
	let inf_2 = InftyList {
		heads : vec![1,2,3,4, 0],
		tail : 0
	}; // just for test
	assert_eq!(inf_1, inf_2);
	
	let inf_1 = InftyList::from(vec![
        vec![   None, Some(1), Some(2)],
        vec![   None],
        vec![Some(5)],
	], vec![]);
	let inf_2 = InftyList::from(vec![
        vec![   None, Some(1),    None],
        vec![   None],
        vec![Some(5)],
	], vec![]);
	assert_ne!(inf_1, inf_2);


}

use std::collections::HashSet;
#[derive(PartialEq)] // !!!!!!!!! I tried to change it
pub struct Diagram(InftyList<usize>);
impl Diagram {
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	fn check(v : &Vec<usize>) {
		if let Err(s) = check_weakly_decreasing(v) {
			panic!(format!("young diagram {}", s))
		}
	}
	
	pub fn from(v : Vec<usize>) -> Diagram {
		Diagram::check(&v);
		Diagram(InftyList::from(v, 0usize))
	}

	/// n(6, 4, 4, 2) = 6 + 4 + 4 + 2 = 16
	pub fn n(&self) -> usize {
		self.0.iter().take_while(|e| {**e > 0}).sum()
	}

	/// |(6, 4, 4, 2)| = 4
	pub fn abs(&self) -> usize {
		self.0.iter().take_while(|e| {**e > 0}).count()
	}

	pub fn transpose(&self) -> Diagram {
		let mut list = InftyList::new();

		let mut i : usize = 0;
		while list.push(self.0.iter().take_while(|e| {**e > i}).count()) {
			i += 1;
		}
		Diagram(list)
	}

	pub fn rows_of_corners(&self) -> HashSet<usize> {
		// self.0.iter().enumerate().tuple_windows::<(_,_)>().filter(|((_, prev), (_, next))| {prev > next}).take_while(|((_, e), _)| {**e > 0}).inspect(|x| {eprintln!("{:?}: e > 0", x)}).map(|((index, _), _)| {index}).collect::<Vec<usize>>()
		let mut rows = HashSet::new();
		let mut iter = self.0.iter().enumerate().peekable();
		let mut this = iter.next();
		let mut next = iter.peek();
		loop {
			if *this.unwrap().1 > *next.unwrap().1 {
				rows.insert(this.unwrap().0);
			} else if *this.unwrap().1 == 0 {
				break;
			}
			this = iter.next();
			next = iter.peek();
		}
		rows
	}
	pub fn rows_below_corners(&self) -> HashSet<usize> {
		self.rows_of_corners().into_iter().map(|x| {x + 1}).collect()
	}
}
#[test]
fn transpose() {
	let diagram = Diagram::from(vec![6,5,3,2,1]);
	assert_eq!(diagram.transpose(), Diagram::from(vec![5,4,3,2,2,1]));
}
#[test]
fn corners() {
	let diagram = Diagram::from(vec![3,2,2,1]);
	assert_eq!(diagram.rows_of_corners(), [0,2,3].iter().cloned().collect::<HashSet<usize>>());
}

impl PartialOrd for Diagram {
	fn partial_cmp(&self, other: &Diagram) -> Option<Ordering> {
		let cmp = self.0.iter().zip(other.0.iter());
		let cmp_l = cmp.clone();
		if self.eq(other) {
			Some(Ordering::Equal)
		} else if cmp.take_while(|(s, _)| {**s == 0}).all(|(s, o)| {*s <= *o}) {
			Some(Ordering::Less)
		} else if cmp_l.take_while(|(_, o)| {**o == 0}).all(|(s, o)| {*s >= *o}) {
			Some(Ordering::Greater)
		} else {
			None
		}
    }
}
#[test]
fn order() {
	let diagram = Diagram::from(vec![5,4,1]);
	let diagram_e = Diagram::from(vec![5,4,1]);
	let diagram_l = Diagram::from(vec![4,4,1]);

	// eprintln!("{}", diagram);
	// eprintln!("{}", diagram_l);
	assert!(diagram_e == diagram);
	assert!(diagram_l <= diagram);
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
	// eprintln!("{}", diagram);
	assert_eq!(format!("{}", diagram), String::from("■ ■ ■ \n■ ■ \n■ \n"));
}

#[derive(PartialEq)]
pub struct SkewDiagram {
	inner : Diagram,
	outer : Diagram,
}
impl SkewDiagram {
	fn check(&self) {
		for (&i, &o) in self.inner.0.iter().take_while(|i| {**i > 0}).zip(self.outer.0.iter()) {
			if i > o {
				panic!("the inner Diagram should be \"inside\" of outer Diagram")
			}
		}
	}

	pub fn from(inner : Diagram, outer : Diagram) -> SkewDiagram {
		let skew_diagram = SkewDiagram {
			inner,
			outer,
		};
		
		skew_diagram.check();
		skew_diagram
	}

	/// the common corners of inner and outer
	pub fn common_corners(&self) -> HashSet<usize> {
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
		Diagram(InftyList::from(vec![2,1], 0)),
		Diagram(InftyList::from(vec![3,2,1], 0)),
	);
	// eprintln!("{}", skew_diagram);
	assert_eq!(format!("{}", skew_diagram), String::from("□ □ ■ \n□ ■ \n■ \n"));
	assert_eq!(format!("{:?}", skew_diagram), String::from("λ: (2, 1, 0...), μ: (3, 2, 1, 0...)"));
}

// -------------------------------------------------------------
#[derive(PartialEq, Debug, Clone)]
pub struct Tableau(InftyList<Vec<Option<usize>>>);
impl Tableau {
	pub fn new() -> Tableau {
		Tableau(InftyList::new())
	}

	pub fn from(v : Vec<Vec<Option<usize>>>) -> Tableau {
		let tableau = Tableau(InftyList::from(v, Vec::new()));
		tableau.check();
		tableau
	}

	/// ! unstale: I used the "heads" attributes
	fn pre_transpose(&self) -> InftyList<Vec<Option<usize>>> {
		let mut v : Vec<Vec<Option<usize>>> = Vec::new();
		for index in 0..self.0.heads[0].len() { // !
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
		InftyList::from(v, Vec::new())
	}

	fn check(&self) {
		self.shape().check();

		for row in self.0.iter().take_while(|e| {**e != Vec::new()}) {
			if let Err(s) = check_weakly_increasing(row) {
				panic!(format!("rows in tableau {}", s))
			}
		}

		for col in self.pre_transpose().iter().take_while(|e| {**e != Vec::new()}) {
			if let Err(s) = check_strictly_increasing(col) {
				panic!(format!("cols in tableau {}", s))
			}
		}
	}

	fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	pub fn shape(&self) -> SkewDiagram {
		SkewDiagram::from(
			Diagram::from(self.0.iter().map(|v| {v.iter().filter(|e| {e.is_none()}).count()}).take_while(|len| {*len > 0}).collect()),
			Diagram::from(self.0.iter().map(|v| {v.iter().count()}).take_while(|len| {*len > 0}).collect()),
		)
	}

	/// return the row_index of the greatest element
	/// pick the rightmost within the equals
	pub fn greatest(&self) -> Option<usize> {
		if self.is_empty() {
			return None;
		}
		let mut gest : Option<usize> = None;
		let mut aim_index = usize::MAX;
		for index in self.shape().outer.rows_of_corners() {
			let tmp = *self.0[index].last().unwrap();
			if tmp > gest {
				aim_index = index;
				gest = tmp;
			} else if tmp == gest && index < aim_index {
				aim_index = index;
			}
		}
		Some(aim_index)
	}
}
#[test]
fn check_skew_tableau() {
	assert!(
		std::panic::catch_unwind(|| {
			Tableau::from(vec![
				vec![   None, Some(1),    None, Some(2)],
				vec![   None, Some(2)],
				vec![Some(2)],
			]);
		}).is_err()
	)
}
#[test]
fn shape() {
	let tableau = Tableau::from(vec![
        vec![   None,    None, Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	
	// eprintln!("{}", tableau.shape());
	assert_eq!(tableau.shape(), SkewDiagram {
		inner : Diagram(InftyList::from(vec![2], 0)),
		outer : Diagram(InftyList::from(vec![3,2,1], 0)),
	});
}

impl Tableau {
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
	/// or `row_bumping`
	pub fn row_insert(&mut self, x : usize) -> usize {
		self.row_bumping(x)
	}

	/// ! unstable: `self.0.get_mut()`
	/// return the value inserted 
	pub fn reverse_bumping(&mut self, row_index : usize) -> usize {
		if let Some(Some(mut bumped)) = self.0[row_index].pop() {
			for row in self.0.iter_mut().take(row_index).rev() { // ?
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

	/// ! unstable: `self.0.get_mut()`
	/// return the index of the removed corner
	pub fn sliding(&mut self, mut row_index : usize) -> usize {
		// eprintln!("{}", self);
		if self.shape().inner.rows_of_corners().contains(&row_index) {
			let mut col_index = self.shape().inner.0[row_index] - 1; // the last one
			// `hole` is self.0[row_index][col_index]
			loop {
				// eprintln!("{}", self);
				// eprintln!("hole: ({}, {})", col_index, row_index);
				let below_one = if let Some(next_row) = self.0.get_mut(row_index + 1) {
					if let Some(e) = next_row.get_mut(col_index) {
						e.take()
					} else {
						None
					}
				} else {
					None
				};
				// eprintln!("below one: {:?}", below_one);
				let right_one = match self.0[row_index].get_mut(col_index + 1) {
					Some(e) => e.take(),
					None => None
				};
				// eprintln!("right one: {:?}", right_one);
				
				if (below_one, right_one) == (None, None) {
					self.0[row_index].pop(); // remove the hole
					// eprintln!("end");
					break row_index;

				} else if (right_one.is_some() && right_one < below_one) || below_one.is_none() {
					self.0[row_index][col_index] = right_one;
					if below_one.is_some() {
						self.0[row_index + 1][col_index] = below_one;
					}
					col_index += 1;
					// eprintln!("move to right");
				} else if (below_one.is_some() && right_one >= below_one) || right_one.is_none(){
					// if they are equal, move to the below one
					self.0[row_index][col_index] = below_one;
					if right_one.is_some() {
						self.0[row_index][col_index + 1] = right_one;
					}
					row_index += 1;
					// eprintln!("move to below");
				} else {
					panic!("Tell me, WTH is the else case?!")
				}
				
			}
		} else {
			panic!("no inner_corner in this row")
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
				// eprintln!("start a new hole at {}", row_index);
				self.sliding(row_index);
			}

		}
	}

	/// ! unstable: `self.0.get_mut()`
	/// note: `reverse` of sliding(1) may not be the reverse_sliding(1), it should be reverse_sliding(sliding(1))
	pub fn reverse_sliding(&mut self, mut row_index : usize) -> usize { // +
		// eprintln!("{}", self);
		if self.shape().outer.rows_below_corners().contains(&row_index) || row_index == 0 {
			let mut col_index = *self.shape().outer.0.iter().nth(row_index).unwrap(); // the one right to the last one
			// `hole` is self.0[row_index][col_index]
			self.0[row_index].push(None);
			loop {
				// eprintln!("{}", self);
				// eprintln!("hole: ({}, {})", col_index, row_index);
				let above_one = if row_index == 0 {
					None
				} else {
					self.0[row_index - 1][col_index].take()
				};
				// eprintln!("above one: {:?}", above_one);
				#[allow(non_snake_case)]
				let left__one = if col_index == 0 {
					None
				} else {
					self.0[row_index][col_index - 1].take()
				};
				// eprintln!("left one: {:?}", left__one);
				
				if (above_one, left__one) == (None, None) {
					// eprintln!("end");
					break row_index;

				} else if (left__one.is_some() && left__one > above_one) || above_one.is_none() {
					self.0[row_index][col_index] = left__one;
					if above_one.is_some() {
						self.0[row_index - 1][col_index] = above_one;
					}
					col_index -= 1;
					// eprintln!("move to left");
				} else if (above_one.is_some() && left__one <= above_one) || left__one.is_none(){
					// if they are equal, move to the below one
					self.0[row_index][col_index] = above_one;
					if left__one.is_some() {
						self.0[row_index][col_index - 1] = left__one;
					}
					row_index -= 1;
					// eprintln!("move to above");
				} else {
					panic!("Tell me, WTH is the else case?!")
				}
				
			}
		} else {
			panic!("no potential common_corners in this row")
		}
	}

	pub fn place(&mut self, row_index : usize, value : usize) {
		self.0[row_index].push(Some(value));
	}
	
	pub fn pop_at(&mut self, row_index : usize) -> usize {
		self.0[row_index].pop().expect("this row is empty, you cannot pop at here").unwrap()
	}
}
#[test]
fn bumping() {
	let mut tableau = Tableau::from(vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	// eprintln!("{}", tableau);
	
	let r1 = tableau.row_bumping(2);
	// eprintln!("{}", tableau);
	assert_eq!(tableau, Tableau::from(vec![
        vec![Some(1), Some(2), Some(2)],
        vec![Some(2), Some(3), Some(3)],
        vec![Some(4)],
	]));

	let r2 = tableau.row_bumping(1);
	// eprintln!("{}", tableau);
	assert_eq!(tableau, Tableau::from(vec![
        vec![Some(1), Some(1), Some(2)],
        vec![Some(2), Some(2), Some(3)],
        vec![Some(3)],
        vec![Some(4)],
	]));
	
	assert_eq!(tableau.reverse_bumping(r2), 1);
	// eprintln!("{}", tableau);
	// assert_eq!(tableau, Tableau::from(vec![
    //     vec![Some(1), Some(2), Some(2)],
    //     vec![Some(2), Some(3), Some(3)],
    //     vec![Some(4)],
	// ]));
	
	assert_eq!(tableau.reverse_bumping(r1), 2);
	// eprintln!("{}", tableau);
	// assert_eq!(tableau, Tableau::from(vec![
    //     vec![Some(1), Some(2), Some(3)],
    //     vec![Some(2), Some(3)],
    //     vec![Some(4)],
	// ]));
}
#[test]
fn sliding() {
	let mut skew_tableau = Tableau::from(vec![
        vec![   None,    None, Some(2)],
        vec![   None, Some(6)],
        vec![Some(5)],
	]);
	// eprintln!("{}", skew_tableau);
	// go right
	let r1 = skew_tableau.sliding(0);
	assert_eq!(skew_tableau, Tableau::from(vec![
        vec![   None, Some(2)],
        vec![   None, Some(6)],
        vec![Some(5)],
	]));

	// go below
	let r2 = skew_tableau.sliding(1);
	assert_eq!(skew_tableau, Tableau::from(vec![
        vec![   None, Some(2)],
        vec![Some(5), Some(6)],
	]));

	assert_eq!(skew_tableau.reverse_sliding(r2), 1);
	assert_eq!(skew_tableau.reverse_sliding(r1), 0);
	assert_eq!(skew_tableau, Tableau::from(vec![
        vec![   None,    None, Some(2)],
        vec![   None, Some(6)],
        vec![Some(5)],
	]));

	let mut skew_tableau = Word(vec![1,2,42,4,5,3,3,4,3,4,2,233,2]).to_skew_tableau();
	// eprintln!("{}", skew_tableau);

	skew_tableau.rect();
	// eprintln!("{}", tableau);
	let tableau = skew_tableau;
	assert_eq!(tableau,Tableau::from(vec![
		vec![Some(1), Some(2), Some(2), Some(2), Some(3), Some(4), Some(233)],
		vec![Some(3), Some(3)],
		vec![Some(4), Some(4)],
		vec![Some(5)],
		vec![Some(42)],
	]));

	
}

impl Mul for Tableau {
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

impl fmt::Display for Tableau {
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
	let tableau = Tableau::from(vec![
		vec![   None, Some(6), Some(7), Some(8)],
		vec![Some(5), Some(7), Some(8)],
		vec![Some(6), Some(8)],
	]);
	// eprintln!("{}", tableau);
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
	pub fn to_skew_tableau(&self) -> Tableau {
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
		Tableau::from(vec)
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
impl Tableau {
	/// ! unstale: iter_finite()
	/// w(T) = from left below side to right and row by row upwards
	pub fn to_word(&self) -> Word { // may be -> Word
		Word(self.0.iter_finite().rev().flatten().filter(|e| {e.is_some()}).map(|e| {e.unwrap()}).collect()) // ! rev
	}
}
#[test]
fn word() {
	let tableau = Tableau::from(vec![
		vec![   None,Some(4),Some(5),Some(6),Some(7)],
		vec![Some(5),Some(7)],
		vec![Some(7)],
	]);
	
	// eprintln!("{}", tableau);
	let word = tableau.to_word();
	assert_eq!(word, Word(vec![7, 5, 7, 4, 5, 6, 7]));
	
	let skew_tableau = word.to_skew_tableau();
	// eprintln!("{}", skew_tableau);
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
fn equivalence_mul() {
	let lhs = Word(vec![1,2,3,4,2,3,4,9,2,3,2,3,4,2,2]);
	let rhs = Word(vec![1,2,3,42,343,464,334,33,2,3,2,5,3,43,2,1]);
	assert_eq!((lhs.clone() * rhs.clone()).to_tableau(), lhs.to_tableau() * rhs.to_tableau());
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
struct Pair(usize, usize);
impl PartialOrd for Pair {
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
impl Ord for Pair {
	fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Pair {
	pub fn from((index, value) : (usize, usize)) -> Pair {
		Pair(index, value)
	}

	pub fn rev(self) -> Pair {
		Pair(self.1, self.0)
	}
}

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