use std::fmt;
use std::ops::{Mul, Deref, DerefMut};

use super::{MathClass};

mod tools;
use tools::VecTail;
use tools::order::{is_strictly_increasing, is_weakly_increasing, replace_greatest_predecessor, replace_least_successor};

#[derive(PartialEq, PartialOrd, Eq, Debug, Clone)]
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
impl Deref for Diagram {
	type Target = VecTail<usize>;
	fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Diagram {
	fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Diagram {
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}
	
	pub fn new() -> Diagram {
		Diagram(VecTail::new())
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
		self.0.iter_finite().sum()
	}

	/// |(6, 4, 4, 2)| = 4
	pub fn abs(&self) -> usize {
		self.0.significant_length()
	}

	/// maybe also named transpose
	pub fn conjugate(&self) -> Diagram {
		let mut list = VecTail::new();
		// println!("{:?}", self);
		// println!("{}", self);
		for i in 0..self.0[0]  {
			// println!("i : {}", i);
			let depth = self.0.iter_finite().take_while(|e| {**e > i}).count();
			list[i] = depth;
		}
		println!("{:?}", list);
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
#[test] fn conjugate() {
	let diagram = Diagram::from(vec![6,6,3,2,1]);
	assert_eq!(diagram.conjugate(), Diagram::from(vec![5,4,3,2, 2,2]));
}
#[test] fn corners() {
	let diagram = Diagram::from(vec![3,2,2,1]);
	assert_eq!(diagram.rows_of_corners(), vec![0,2,3]);
}

impl fmt::Display for Diagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for &len in self.0.iter_finite() {
			writeln!(f, "{}", "■ ".repeat(len))?;
		}
		write!(f, "")
    }
}
#[test] fn display_diagram() {
	let diagram = Diagram::from(vec![3,2,1]);
	// println!("{}", diagram);
	assert_eq!(format!("{}", diagram), String::from("■ ■ ■ \n■ ■ \n■ \n"));
}
#[test] fn order() {
	let diagram = Diagram::from(vec![5,4,1]);
	let diagram_e = Diagram::from(vec![5,4,1]);
	let diagram_l = Diagram::from(vec![4,4,1]);

	// println!("{}", diagram);
	// println!("{}", diagram_l);
	assert!(diagram_e == diagram);
	assert!(diagram_l <= diagram);
}

#[derive(PartialEq, Eq)]
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
#[test] fn display_skew_diagram() {
	let skew_diagram = SkewDiagram::from(
		Diagram(VecTail::from(vec![2,1], 0)),
		Diagram(VecTail::from(vec![3,2,1], 0)),
	);
	// println!("{}", skew_diagram);
	assert_eq!(format!("{}", skew_diagram), String::from("□ □ ■ \n□ ■ \n■ \n"));
}

impl Mul for SkewDiagram {
	type Output = SkewDiagram;
    /// simply put the right skew Diagram on the right above block of left one
	/// i.e. λ * μ (page 60) = 
	/// □ □ ■ ■
	/// □ □ ■
	/// □ ■
	/// ■ ■
    fn mul(self, rhs: Self) -> SkewDiagram {
		let delta_height = rhs.outer.0.significant_length() - rhs.inner.0.significant_length();
		let length = self.outer.0[0];
		SkewDiagram::from(
			Diagram::from(rhs.inner.0.into_iter_finite().map(|e| {
				length + e
			}).chain(vec![length ; delta_height].into_iter()).chain(self.inner.0.into_iter_finite()).collect()),

			Diagram::from(rhs.outer.0.into_iter_finite().map(|e| {
				length + e
			}).chain(self.outer.0.into_iter_finite()).collect()),
		)
    }
}
impl Diagram {
	pub fn to_skew_diagram(&self) -> SkewDiagram {
		SkewDiagram::from(Diagram::new(), self.clone())
	}
}
impl Mul for Diagram {
	type Output = SkewDiagram;
    /// simply put the right Diagram on the right above block of left one
	/// i.e. λ * μ (page 60) = 
	/// □ □ ■ ■
	/// □ □ ■
	/// □ ■
	/// ■ ■
    fn mul(self, rhs: Self) -> SkewDiagram {
		self.to_skew_diagram() * rhs.to_skew_diagram()
    }
}
#[test] fn mul_skew_diagram() {
	let sd_1 = SkewDiagram::from(
		Diagram(VecTail::from(vec![2], 0)),
		Diagram(VecTail::from(vec![3,2,1], 0)),
	);

	let sd_2 = SkewDiagram::from(
		Diagram(VecTail::from(vec![1], 0)),
		Diagram(VecTail::from(vec![3,2], 0))
	);

	assert_eq!(sd_1.inner.clone() * sd_2.inner.clone(), SkewDiagram::from(
		Diagram(VecTail::from(vec![2], 0)),
		Diagram(VecTail::from(vec![3,2], 0)),
	));

	// println!("{}", sd_1 * sd_2);
	assert_eq!(sd_1 * sd_2, SkewDiagram::from(
		Diagram(VecTail::from(vec![4,3,2], 0)),
		Diagram(VecTail::from(vec![6,5,3,2,1], 0)),
	));
}

// -------------------------------------------------------------
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Filling(VecTail<Vec<usize>>);
impl MathClass for Filling {
	fn check(&self) -> Result<(), String> {
		self.shape().check()?;
		Ok(())
	}
}
impl Deref for Filling {
	type Target = VecTail<Vec<usize>>;
	fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Filling {
	fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Filling {
	pub fn new() -> Filling {
		Filling(VecTail::new())
	}

	pub fn from(v : Vec<Vec<usize>>) -> Filling {
		let filling = Filling(VecTail::from(v, Vec::new()));

		if let Err(s) = filling.check() {
			panic!("Filling: {}", s);
		}
		filling
	}

	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	pub fn transpose(&self) -> Filling {
		Filling(self.pre_transpose())
	}

	pub fn shape(&self) -> Diagram {
		Diagram::from(self.0.iter_finite().map(|v| {v.iter().count()}).collect())
	}
}
impl Filling { // convert -- similiarly
	pub fn to_numbering(self) -> Numbering {
		let numbering = Numbering {
			refer : self
		};
		numbering.check().unwrap();
		numbering
	}

	pub fn to_tableau(self) -> Tableau {
		let tableau = Tableau {
			refer : self
		};
		tableau.check().unwrap();
		tableau
	}
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Numbering {
	refer : Filling,
}
use std::collections::HashSet;
impl MathClass for Numbering {
	fn check(&self) -> Result<(), String> {
		self.refer.check()?;

		if self.0.iter_finite().flatten().collect::<HashSet<_>>().len() != self.shape().n() {
			Err("Numbering should have distinct entries".into())
		} else {
			Ok(())
		}
	}
}
#[test] fn numbering() {
	let numbering = Numbering {
		refer : Filling::from(vec![vec![1,2,3,4], vec![2,3,4]])
	};
	assert!(numbering.check().is_err());
	let numbering = Numbering {
		refer : Filling::from(vec![vec![1,2,3,4], vec![5,6,7]])
	};
	assert!(numbering.check().is_ok());
}
impl Deref for Numbering {
	type Target = Filling;
	fn deref(&self) -> &Self::Target {
        &self.refer
    }
}
impl DerefMut for Numbering {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.refer
    }
}
impl Numbering {
	pub fn new() -> Numbering {
		Numbering {
			refer : Filling::new()
		}
	}

	pub fn from(v : Vec<Vec<usize>>) -> Numbering {
		let numbering = Numbering {
			refer : Filling::from(v)
		};

		if let Err(s) = numbering.check() {
			panic!("Numbering: {}", s);
		}
		numbering
	}

	// ? to complicated 
	// ! direct copied from below
	pub fn transpose(&self) -> Numbering {
		Numbering {
			refer : self.refer.transpose()
		}
	}
}
impl Numbering { // convert -- similiarly
	pub fn to_filling(self) -> Filling {
		self.refer
	}

	pub fn to_tableau(self) -> Tableau {
		self.to_filling().to_tableau()
	}
	
	pub fn to_standard_tableau(self) -> StandardTableau {
		self.to_filling().to_tableau().to_standard_tableau()
	}
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Tableau {
	refer : Filling,
}
impl Deref for Tableau {
	type Target = Filling;
	fn deref(&self) -> &Self::Target {
        &self.refer
    }
}
impl DerefMut for Tableau {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.refer
    }
}
impl MathClass for Tableau {
	// ? too complicated
	// ! direct copied from below
	fn check(&self) -> Result<(), String> {
		self.refer.check()?;

		for row in self.0.iter_finite() {
			if let Err(s) = is_weakly_increasing(row) {
				return Err(format!("rows in tableau {}", s));
			}
		}
		for col in self.pre_transpose().iter_finite() {
			if let Err(s) = is_strictly_increasing(col) {
				return Err(format!("cols in tableau {}", s))
			}
		}
		Ok(())
	}
}
impl Tableau {
	pub fn new() -> Tableau {
		Tableau {
			refer : Filling::new()
		}
	}

	pub fn from(v : Vec<Vec<usize>>) -> Tableau {
		let tableau = Tableau{
			refer : Filling::from(v)
		};

		if let Err(s) = tableau.check() {
			panic!("Tableau: {}", s);
		}
		tableau
	}

	/// return the row_index of the greatest element
	/// pick the rightmost within the equals
	pub fn greatest_row(&self) -> Option<usize> {
		// println!("self: {:?}", self);
		// println!("{:?}", self.is_empty());
		if self.is_empty() {
			None
		} else {
			let mut gest : usize = usize::MIN;
			let mut aim_index = usize::MAX;
			for index in self.shape().rows_of_corners() {
				let tmp = *self.0[index].last().unwrap();
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

	pub fn greatest(&self) -> usize {
		if let Some(row) = self.greatest_row() {
			*self.0[row].last().unwrap()
		} else {
			usize::MIN
		}
	}

	pub fn content_0(&self) -> Vec<usize> {
		let mut v = VecTail::new();
		for line in self.0.iter_finite() {
			let mut iter = line.iter().peekable();
			for i in 0..=*line.last().unwrap() {
				while Some(&&i) == iter.peek() {
					v[i] += 1;
					iter.next();
				}
			}
		}
		v.into_iter_finite().collect()
	}
		pub fn weight_0(&self) -> Vec<usize> {self.content_0()}
		pub fn type_0(&self) -> Vec<usize> {self.content_0()}

	pub fn content_1(&self) -> Vec<usize> {
		if Some(&0) == self.0[0].first() {
			panic!("it seems you use a tableau start from 0, try content_0 please");
		} else {
			let mut v = VecTail::new();
			for line in self.0.iter_finite() {
				let mut iter = line.iter().peekable();
				for i in 1..=*line.last().unwrap() { // * 1
					while Some(&&i) == iter.peek() {
						v[i - 1] += 1; // * 1
						iter.next();
					}
				}
			}
			v.into_iter_finite().collect() 
		}
	}
		pub fn weight_1(&self) -> Vec<usize> {self.content_1()}
		pub fn type_1(&self) -> Vec<usize> {self.content_1()}
}
#[test] fn greatest_row() {
	let mut tableau = Tableau::from(vec![
        vec![1, 2, 3],
        vec![2, 3],
        vec![4],
	]);
	assert_eq!(tableau.greatest_row(), Some(2));
	tableau.pop_at_row(2);
	// println!("{}", tableau);
	assert_eq!(tableau.greatest_row(), Some(0));
	tableau.pop_at_row(0);
	// println!("{}", tableau);
	assert_eq!(tableau.greatest_row(), Some(1));
}
impl Tableau { // convert -- similiarly
	pub fn to_filling(self) -> Filling {
		self.refer
	}
	
	/// panic if it do not satisfies the criteria of StandardTableau
	pub fn to_standard_tableau(self) -> StandardTableau {
		StandardTableau::from(self.refer.0.into_iter_finite().collect()) 
	}

	pub fn to_skew_tableau(self) -> SkewTableau {
		SkewTableau::from(
			self.refer.0.into_iter_finite().map(|v| {
				v.into_iter().map(|e| Some(e)).collect()
			}).collect()
		)
	}
}
#[test] fn content() {
	let t = Tableau::from(vec![
        vec![1, 2, 3],
        vec![2, 3],
        vec![4],
	]);
	// println!("{:?}", t.content_0());
	assert_eq!(vec![0,1,2,2,1], t.content_0());
	assert_eq!(vec![1,2,2,1], t.content_1());

	assert_eq!(t.content_0().into_iter().sum::<usize>(), t.shape().n());
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
		self.0[row_index].push(bumped);
		row_index
	}
		pub fn row_insert(&mut self, x : usize) -> usize {
			self.row_bumping(x)
		}

	/// ? to complicated
	/// return the value inserted 
	pub fn reverse_bumping(&mut self, row_index : usize) -> usize {
		if let Some(mut bumped) = self.0[row_index].pop() {
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
}
#[test] fn bumping() {
	let mut tableau = Tableau::from(vec![
        vec![1, 2, 3],
        vec![2, 3],
        vec![4],
	]);
	// println!("{}", tableau);
	
	let r1 = tableau.row_bumping(2);
	assert_eq!(r1, 1);
	// println!("{}", tableau);
	assert_eq!(tableau, Tableau::from(vec![
        vec![1, 2, 2],
        vec![2, 3, 3],
        vec![4],
	]));

	let r2 = tableau.row_bumping(1);
	assert_eq!(r2, 3);
	// println!("{}", tableau);
	assert_eq!(tableau, Tableau::from(vec![
        vec![1, 1, 2],
        vec![2, 2, 3],
        vec![3],
        vec![4],
	]));
	
	assert_eq!(tableau.reverse_bumping(r2), 1);
	// println!("{}", tableau);
	assert_eq!(tableau, Tableau::from(vec![
        vec![1, 2, 2],
        vec![2, 3, 3],
        vec![4],
	]));
	
	assert_eq!(tableau.reverse_bumping(r1), 2);
	// println!("{}", tableau);
	assert_eq!(tableau, Tableau::from(vec![
        vec![1, 2, 3],
        vec![2, 3],
        vec![4],
	]));
}

impl Tableau {
	/// simply put a element on this row
	pub fn place_at_row(&mut self, row_index : usize, value : usize) {
		self[row_index].push(value);
	}
	
	pub fn pop_at_row(&mut self, row_index : usize) -> usize {
		if let Some(value) = self[row_index].pop() {
			self.strip(); // ?
			value
		} else {
			panic!("this row is empty, you cannot pop at here");
		}
	}
}
#[test] fn place_and_pop() {
	let mut tableau = Tableau::from(vec![
        vec![1, 2, 3],
        vec![2, 3],
        vec![4],
	]);
	tableau.pop_at_row(2);
	// println!("{}", tableau);
	tableau.pop_at_row(1);
	// println!("{}", tableau);
}


impl fmt::Display for Tableau {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for v in self.0.iter_finite() {
			for e in v.iter() {
				write!(f, "{} ", e)?;
			}
			writeln!(f, "")?;
		}
		write!(f, "")
    }
}
#[test] fn display_tableau() {
	let tableau = Tableau::from(vec![
		vec![2, 6, 7, 8],
		vec![5, 7, 8],
		vec![6, 8],
	]);
	// println!("{}", tableau);
	assert_eq!(format!("{}", tableau), String::from("2 6 7 8 \n5 7 8 \n6 8 \n"));
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct StandardTableau {
	refer : Tableau
}
impl MathClass for StandardTableau {
	// ? too complicated
	fn check(&self) -> Result<(), String> {
		self.shape().check()?;

		for col in self.pre_transpose().iter_finite() {
			if let Err(s) = is_strictly_increasing(col) {
				return Err(format!("cols in tableau {}", s))
			}
		}
		Ok(())
	}
}
impl Deref for StandardTableau {
	type Target = Tableau;
	fn deref(&self) -> &Self::Target {
        &self.refer
    }
}
impl DerefMut for StandardTableau {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.refer
    }
}
#[test] fn standard_tableau_content() {
	assert_eq!(StandardTableau::from(vec![
		vec![1,2,3],
		vec![4,5,6]
	]).content_1(), vec![1 ; 6]);
}
impl StandardTableau {
	pub fn new() -> StandardTableau {
		StandardTableau {
			refer : Tableau::new()
		}
	}

	pub fn from(v : Vec<Vec<usize>>) -> StandardTableau {
		let tableau = StandardTableau {
			refer : Tableau::from(v)
		};

		if let Err(s) = tableau.check() {
			panic!("StandardTableau: {}", s);
		}
		tableau
	}

	pub fn transpose(&self) -> StandardTableau {
		StandardTableau {
			refer : Tableau {
				refer : Filling(self.pre_transpose())
			}
		}
	}
}
impl StandardTableau { // convert -- similiarly
	pub fn to_tableau(self) -> Tableau {
		self.refer
	}

	pub fn to_skew_tableau(self) -> SkewTableau {
		self.to_tableau().to_skew_tableau()
	}

	pub fn to_filling(self) -> Filling {
		self.to_tableau().to_filling()
	}

	pub fn to_numbering(self) -> Numbering {
		self.to_filling().to_numbering()
	}
}
impl fmt::Display for StandardTableau {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.refer.fmt(f)
    }
}

// ------------------------------------------------------
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SkewTableau(VecTail<Vec<Option<usize>>>);
impl MathClass for SkewTableau {
	// ? too complicated
	fn check(&self) -> Result<(), String> {
		self.shape().check()?;

		for row in self.0.iter_finite() {
			if let Err(s) = is_weakly_increasing(&row.iter().skip_while(|e| e.is_none()).collect()) {
				return Err(format!("rows in tableau {}", s));
			}
		}
		for col in self.pre_transpose().iter_finite() {
			if let Err(s) = is_strictly_increasing(&col.iter().skip_while(|e| e.is_none()).collect()) {
				return Err(format!("cols in tableau {}", s))
			}
		}
		Ok(())
	}
}
#[test] fn check_skew_tableau() {
	assert!(std::panic::catch_unwind(|| {
		SkewTableau::from(vec![
			vec![   None,    None, Some(1), Some(2)],
			vec![   None, Some(2)],
			vec![Some(2)],
		]);
	}).is_ok());
	assert!(std::panic::catch_unwind(|| {
		SkewTableau::from(vec![
			vec![   None, Some(1),    None, Some(2)],
			vec![Some(1), Some(2)],
			vec![   None, Some(2)],
			vec![Some(2)],
		]);
	}).is_err());
	assert!(std::panic::catch_unwind(|| {
		SkewTableau::from(vec![
			vec![   None, Some(2)],
			vec![Some(5), Some(6)]
		]);
	}).is_ok())
}
impl Deref for SkewTableau {
	type Target = VecTail<Vec<Option<usize>>>;
	fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for SkewTableau {
	fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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

	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	pub fn content_0(&self) -> Vec<usize> {
		unimplemented!()
	}
		pub fn weight_0(&self) -> Vec<usize> {self.content_0()}
		pub fn type_0(&self) -> Vec<usize> {self.content_0()}

	pub fn content_1(&self) -> Vec<usize> {
		unimplemented!()
	}
		pub fn weight_1(&self) -> Vec<usize> {self.content_1()}
		pub fn type_1(&self) -> Vec<usize> {self.content_1()}


	pub fn shape(&self) -> SkewDiagram {
		SkewDiagram::from(
			Diagram::from(self.0.iter_finite().map(|v| {v.iter().take_while(|e| {e.is_none()}).count()}).collect()),
			Diagram::from(self.0.iter_finite().map(|v| {v.iter().count()}).collect()),
		)
	}
}
impl SkewTableau { // convert -- similiarly
	pub fn to_tableau(self) -> Tableau {
		Tableau::from(
			self.0.into_iter_finite().map(|v| {
				v.into_iter().map(|e| e.expect("this is not a tableau, try .rect() please")).collect()
			}).collect()
		)
	}
}
#[test] fn conversion_skew_or_not() {
	assert!(std::panic::catch_unwind(|| {
		SkewTableau::from(vec![
			vec![   None, Some(1), Some(2)],
			vec![   None, Some(2)],
			vec![Some(2)],
		]).to_tableau();
	}).is_err());

	assert_eq!(
		SkewTableau::from(vec![
			vec![Some(1), Some(2)],
			vec![Some(2)],
		]).to_tableau(),
		Tableau::from(vec![
			vec![1, 2],
			vec![2],
		])
	);

	assert_eq!(
		SkewTableau::from(vec![
			vec![Some(1), Some(2)],
			vec![Some(2)],
		]),
		Tableau::from(vec![
			vec![1, 2],
			vec![2],
		]).to_skew_tableau()
	);
}
#[test] fn shape() {
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
#[test] fn content_skew_tableau() {
	// + content
	let t = SkewTableau::from(vec![
        vec![   None, Some(1), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);
	// println!("{:?}", t.content_0());
	assert_eq!(vec![0,1,1,2,1], t.content_0());
	assert_eq!(vec![1,1,2,1], t.content_1());

	// assert_eq!(t.content_0().into_iter().sum::<usize>(), t.shape().n());

	let t = SkewTableau::from(vec![
        vec![   None, Some(0), Some(3)],
        vec![Some(2), Some(3)],
        vec![Some(4)],
	]);

	assert!(
		std::panic::catch_unwind(|| {
			t.content_1()
		}).is_err()
	)
}

impl SkewTableau {
	/// return the index of the removed corner
	pub fn sliding(&mut self, mut row_index : usize) -> usize {
		println!("{}", self);
		if self.shape().inner.rows_of_corners().contains(&row_index) {
			let mut col_index = self.shape().inner.0[row_index] - 1; // the last one
			// * `hole` is self.0[row_index][col_index]
			loop {
				println!("{}", self);
				println!("hole: ({}, {})", col_index, row_index);
				let south_one = {
					let next_row = &mut self.0[row_index + 1];
					if let Some(e) = next_row.get_mut(col_index) {
						e.take()
					} else {
						None
					}
				};
				println!("below one: {:?}", south_one);
				let east_one = match self.0[row_index].get_mut(col_index + 1) {
					Some(e) => e.take(),
					None => None
				};
				println!("right one: {:?}", east_one);
				
				if (south_one, east_one) == (None, None) {
					self.0[row_index].pop(); // remove the hole
					println!("end");
					println!("{}", self);
					break row_index;

				} else if (east_one.is_some() && east_one < south_one) || south_one.is_none() {
					self.0[row_index][col_index] = east_one;
					if south_one.is_some() {
						self.0[row_index + 1][col_index] = south_one;
					}
					col_index += 1;
					println!("move to right");
				} else if (south_one.is_some() && east_one >= south_one) || east_one.is_none(){
					// if they are equal, move to the below one
					self.0[row_index][col_index] = south_one;
					if east_one.is_some() {
						self.0[row_index][col_index + 1] = east_one;
					}
					row_index += 1;
					println!("move to below");
				} else {
					panic!("Tell me, WTH is the else case?!")
				}
				
			}
		} else {
			panic!("no inner corners in this row")
		}
	}

	/// Rect(S)
	/// jeu de taquin
	/// a rectification (redressement) of a skew tableau S
	/// return a Tableau
	pub fn rect(mut self) -> Tableau {
		// do until self.shape().inner is empty
		// I'll repeat on last non-empty inner line, which is always a corner
		let list = self.shape().inner.0;
		for row_index in (0..list.significant_length()).rev() {
			for _ in 0..list[row_index] {
				// println!("start a new hole at {}", row_index);
				self.sliding(row_index);
			}
		}

		self.to_tableau()
	}

	/// note: `reverse` of sliding(1) may not be the reverse_sliding(1), it should be reverse_sliding(sliding(1))
	/// return the row of new box
	pub fn reverse_sliding(&mut self, mut row_index : usize) -> usize {
		// println!("{}", self);
		if self.shape().outer.rows_below_corners().contains(&row_index) || row_index == 0 {
			let mut col_index = *self.shape().outer.0.iter().nth(row_index).unwrap(); // the one right to the last one
			// `hole` is self.0[row_index][col_index]
			self.0[row_index].push(None);
			loop {
				// println!("{}", self);
				// println!("hole: ({}, {})", col_index, row_index);
				let north_one = if row_index == 0 {
					None
				} else {
					self.0[row_index - 1][col_index].take()
				};
				// println!("above one: {:?}", north_one);

				let west_one = if col_index == 0 {
					None
				} else {
					self.0[row_index][col_index - 1].take()
				};
				// println!("left one: {:?}", west_one);
				
				if (north_one, west_one) == (None, None) {
					// println!("end");
					break row_index;

				} else if (west_one.is_some() && west_one > north_one) || north_one.is_none() {
					self.0[row_index][col_index] = west_one;
					if north_one.is_some() {
						self.0[row_index - 1][col_index] = north_one;
					}
					col_index -= 1;
					// println!("move to left");
				} else if (north_one.is_some() && west_one <= north_one) || west_one.is_none(){
					// if they are equal, move to the below one
					self.0[row_index][col_index] = north_one;
					if west_one.is_some() {
						self.0[row_index][col_index - 1] = west_one;
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
#[test] fn sliding() {
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

	assert_eq!(SkewTableau::from(vec![
			vec![   None,    None,     None,    None,    None,    None,     None,    None,    None,    None,    None,     None, Some(2)],
			vec![   None,    None,     None,    None,    None,     None,    None,    None,    None,    None, Some(2), Some(233)],
			vec![   None,    None,     None,    None,    None,     None,    None,    None, Some(3), Some(4)],
			vec![   None,    None,     None,    None,    None, Some(3), Some(3), Some(4)],
			vec![   None,    None,     None, Some(4), Some(5)],
			vec![Some(1), Some(2), Some(42)]
		]).rect(),
		Tableau::from(vec![
			vec![1, 2, 2, 2, 3, 4, 233],
			vec![3, 3],
			vec![4, 4],
			vec![5],
			vec![42],
	]));
	// println!("{}", tableau);

}

impl Mul for SkewTableau {
	type Output = Self;
	/// simply put the right skew tableau on the right above block of left one
	/// i.e. T * U (in page 15) = 
	/// □ □ 2 3
	/// □ □ 3
	/// 2 2
	/// 2 2
    fn mul(self, rhs: Self) -> Self {
		let l = self.0[0].len();
		SkewTableau::from(
			rhs.0.into_iter_finite().map(|v| {
				[vec![None; l], v].concat()
			}).chain(self.0.into_iter_finite()).collect()
		)
    }
}
#[test] fn mul_skew_tableau() {
	let st1 = SkewTableau::from(vec![
		vec![Some(1), Some(2), Some(3)],
		vec![Some(2)],
	]);
	let st2 = SkewTableau::from(vec![
		vec![Some(1), Some(3)],
		vec![Some(2), Some(4)],
	]);

	// println!("{}", st1 * st2);
	assert_eq!(st1.clone().to_tableau() * st2.clone().to_tableau(), (st1 * st2).rect());

}

impl fmt::Display for SkewTableau {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for v in self.0.iter_finite() {
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
#[test] fn display_skew_tableau() {
	let tableau = SkewTableau::from(vec![
		vec![   None, Some(6), Some(7), Some(8)],
		vec![Some(5), Some(7), Some(8)],
		vec![Some(6), Some(8)],
	]);
	// println!("{}", tableau);
	assert_eq!(format!("{}", tableau), String::from("□ 6 7 8 \n5 7 8 \n6 8 \n"));
}