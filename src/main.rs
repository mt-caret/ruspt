mod lib;
use lib::*;

fn main() {
    println!("Path tracing renderer: edupt");

    //render(640, 480, 4, 2);
    render(640, 480, 128, 2);
}
