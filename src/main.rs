#[allow(unused_imports)]
use young_tableaux::{
	Diagram, SkewDiagram,
	SkewTableau,
	Word, TwoRowedArray
};

fn main() {
	assert_eq!(TwoRowedArray::from_two_arrays(vec![1,2,3,4,2,4,4,2], vec![3,4,2,5,4,2,3,2]), TwoRowedArray::from_pairs(
		vec![(1, 3),(2, 4),(3, 2),(4, 5),(2, 4),(4, 2),(4, 3),(2, 2)],
	));
}

// fn main() {
// 	let mut tableau = Tableau::from(vec![
//         vec![Some(1), Some(2), Some(3)],
//         vec![Some(2), Some(3)],
//         vec![Some(4)],
// 	]);
// 	assert_eq!(tableau.greatest_row(), Some(2));
// 	println!("{}", tableau);
// 	tableau.pop_at(2);
// 	println!("{}", tableau);
// 	assert_eq!(tableau.greatest_row(), Some(1));
// 	println!("{}", tableau);
// 	tableau.pop_at(1);
// 	assert_eq!(tableau.greatest_row(), Some(0));
// }

// pub fn greatest(v : &Vec<Option<usize>>) -> Option<usize> {
// 	if v.is_empty() {
// 		return None;
// 	}
// 	let mut gest : Option<usize> = None;
// 	let mut aim_index = usize::MAX;
// 	for index in 0..v.len() {
// 		let tmp = v[index];
// 		if tmp >= gest {
// 			aim_index = index;
// 			gest = tmp;
// 		} else if tmp == gest && index < aim_index {
// 			aim_index = index;
// 		}
// 	}
// 	Some(aim_index)
// }

// fn main() {
// 	let mut v : Vec<Option<usize>> = vec![Some(3), Some(3), Some(4)];
// 	assert_eq!(greatest(&v), Some(2));
// 	v.pop();
// 	assert_eq!(greatest(&v), Some(1));
// }