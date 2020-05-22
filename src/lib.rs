use std::fmt;

#[allow(dead_code)]
mod tools {
	use std::cmp::PartialOrd;
	pub fn check_weakly_decreasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		for index in 0..(v.len() - 1) {
			if v[index + 1] > v[index] {
				return Err("should be weakly deceasing");
			}
		}
		Ok(())
	}
	pub fn check_weakly_increasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		for index in 0..(v.len() - 1) {
			if v[index + 1] < v[index] {
				return Err("should be weakly inceasing");
			}
		}
		Ok(())
	}
	
	pub fn check_strictly_decreasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		for index in 0..(v.len() - 1) {
			if v[index + 1] >= v[index] {
				return Err("should be strictly deceasing");
			}
		}
		Ok(())
	}
	pub fn check_strictly_increasing<T : PartialOrd>(v : &Vec<T>) -> Result<(), &'static str> {
		for index in 0..(v.len() - 1) {
			if v[index + 1] <= v[index] {
				return Err("should be strictly inceasing");
			}
		}
		Ok(())
	}
}
use tools::*;


pub struct Diagram(Vec<usize>);
impl Diagram {
	fn check(&self) {
		if let Err(s) = check_weakly_decreasing(&self.0) {
			panic!(format!("young diagram {}", s))
		}
	}
	
	pub fn from(v : Vec<usize>) -> Diagram {
		let lam = Diagram(v);
		lam.check();
		lam
	}

	/// n(6, 4, 4, 2) = 6 + 4 + 4 + 2 = 16
	pub fn n(&self) -> usize {
		self.0.iter().sum()
	}

	/// |(6, 4, 4, 2)| = 4
	pub fn abs(&self) -> usize {
		self.0.iter().count()
	}
}
impl fmt::Debug for Diagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self.0).replace("[", "(").replace("]", ")"))
    }
}
impl fmt::Display for Diagram {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for &len in self.0.iter() {
			writeln!(f, "{}", "■ ".repeat(len))?;
		}
		write!(f, "")
        // write!(f, "{}", format!("{:?}", self.0).replace("[", "(").replace("]", ")"))
    }
}

pub struct Tableaux(Vec<Vec<Option<usize>>>);
impl Tableaux {
	fn check(&self) {
		self.shape().check();

		for row in self.0.iter() {
			if let Err(s) = check_weakly_increasing(row) {
				panic!(format!("rows {}", s))
			}
		}

		for col in self.cols() {
			if let Err(s) = check_strictly_increasing(&col) {
				panic!(format!("cols {}", s))
			}
		}
	}
	fn cols(&self) -> Vec<Vec<Option<usize>>> {
		// + refine
		let mut v : Vec<Vec<Option<usize>>> = Vec::new();
		for index in 0..self.0[0].len() {
			let mut tmp_v = Vec::new();
			for row in self.0.iter() {
				if index < row.len() {
					tmp_v.push(row[index]);
				} else {
					break;
				}
			}
			v.push(tmp_v);
		}
		v
	}

	pub fn take(&mut self, x : usize, y : usize) -> Result<Option<usize>, &'static str> {
		if let Some(e) = self.0[y][x] {
			return Ok(e);
		}
		Err("no such a cell")
	}

	pub fn from(v : Vec<Vec<Option<usize>>>) -> Tableaux {
		let tableaux = Tableaux(v);
		tableaux.check();
		tableaux
	}

	pub fn shape(&self) -> Diagram {
		Diagram(self.0.iter().map(|v| {v.iter().count()}).collect())
	}
}

impl Tableaux { // operations
	// pub fn bumpping(&self, )
}

impl fmt::Display for Tableaux {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for v in self.0.iter() {
			for e in v.iter() {
				if let Some(e) = e {
					write!(f, "{} ", e)?;
				} else {
					write!(f, "  ")?;
				}
			}
			writeln!(f, "")?;
		}
		write!(f, "")
    }
}