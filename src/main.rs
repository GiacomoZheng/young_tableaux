fn main() {}
/*
#[allow(unused_imports)]
use young_tableaux::{
	Diagram, SkewDiagram, // + Partition
	Filling, Numbering, 
	Tableau, StandardTableau, SkewTableau,
	Word, // + ReverseLatticeWord
		TwoRowedArray, // + Permutation
	TableauPair,
	Matrix, BallMatrix
};

#[allow(unused_variables)]
#[allow(non_snake_case)]
fn main() {
	let A = Matrix::from(vec![
		1, 2, 1,
		2, 2, 0,
		1, 0, 1,
	], 3, 3);

	println!("{}", A.to_tableau_pair_1());
	println!("{}", A);
	println!("{}", A.to_new_matrix());
	println!("{}", A.to_new_matrix().to_new_matrix());
	println!("{}", A.to_new_matrix().to_new_matrix().to_new_matrix());


	let v : Vec<u32> = vec![1,2,3];
	// is the same as
	let v = {
		let mut tmp_v : Vec<u32> = Vec::new();
		tmp_v.push(1);
		tmp_v.push(2);
		tmp_v.push(3);
		tmp_v
	};
}
*/