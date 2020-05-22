// use young_tableaux::Diagram;
use young_tableaux::Tableaux;

fn main() {
    // println!("Hello, world!");
    // let lambda = Diagram::from(vec![3,2,1]);
    // println!("{}", lambda);
    // println!("{}", lambda.n());
    // println!("{}", lambda.abs());

    let tableaux = Tableaux::from(vec![
        vec![1,2,3].into_iter().map(|i| {Some(i)}).collect(),
        vec![2,3].into_iter().map(|i| {Some(i)}).collect(),
        vec![4].into_iter().map(|i| {Some(i)}).collect(),
    ]);
    println!("{}", tableaux);
    println!("{}", tableaux.shape());
    // println!("{}", tableaux);
}
