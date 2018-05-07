extern crate rayon;
use rayon::prelude::*;

use std::ops;

type Seed = [u32; 4];

pub fn next(seed: Seed) -> (u32, Seed) {
    let t = seed[0] ^ (seed[0] << 11);
    (
        seed[3],
        [
            seed[1],
            seed[2],
            seed[3],
            (seed[3] ^ (seed[3] >> 19)) ^ (t ^ (t >> 8)),
        ],
    )
}

pub fn next01(seed: Seed) -> (f64, Seed) {
    let (res, next_seed) = next(seed);
    (res as f64 / ::std::u32::MAX as f64, next_seed)
}

pub fn init_xorshift(seed: u32) -> Seed {
    let mut s = seed;
    let mut ret = [0; 4];
    for i in 0..4 {
        s = 1812433253 * (s ^ (s >> 30)) + i as u32 + 1;
        ret[i] = s;
    }
    ret
}

pub fn clamp(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        if x > 1.0 {
            1.0
        } else {
            x
        }
    }
}

pub fn to_u8(x: f64) -> u8 {
    (clamp(x).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

pub fn save_ppm_file(filename: &str, image: &[Color], width: usize, height: usize) {
    use std::fs;
    use std::io::{BufWriter, Write};
    assert_eq!(image.iter().len(), width * height);
    let mut f = BufWriter::new(fs::File::create(filename).expect("File creation failed."));
    write!(f, "P3\n{} {}\n{}\n", width, height, 255).expect("Write failed.");
    image.iter().for_each(|pixel| {
        write!(
            f,
            "{} {} {} ",
            to_u8(pixel.0),
            to_u8(pixel.1),
            to_u8(pixel.2)
        ).expect("Write failed.");
    });
}

#[derive(Debug, Clone, Copy)]
pub struct V(f64, f64, f64);

impl ops::Add<V> for V {
    type Output = V;
    fn add(self, r: V) -> V {
        V(self.0 + r.0, self.1 + r.1, self.2 + r.2)
    }
}

impl ops::Sub<V> for V {
    type Output = V;
    fn sub(self, r: V) -> V {
        V(self.0 - r.0, self.1 - r.1, self.2 - r.2)
    }
}

impl ops::Mul<f64> for V {
    type Output = V;
    fn mul(self, x: f64) -> V {
        V(self.0 * x, self.1 * x, self.2 * x)
    }
}

impl ops::Mul<V> for V {
    type Output = V;
    fn mul(self, x: V) -> V {
        V(self.0 * x.0, self.1 * x.1, self.2 * x.2)
    }
}

impl ops::Div<f64> for V {
    type Output = V;
    fn div(self, x: f64) -> V {
        V(self.0 / x, self.1 / x, self.2 / x)
    }
}

impl V {
    pub fn len_sq(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }
    pub fn len(&self) -> f64 {
        self.len_sq().sqrt()
    }
    pub fn normalize(self) -> V {
        self * (1.0 / self.len())
    }
    pub fn dot(&self, x: &V) -> f64 {
        self.0 * x.0 + self.1 * x.1 + self.2 * x.2
    }
    pub fn cross(&self, x: &V) -> V {
        V(
            self.1 * x.2 - self.2 - x.1,
            self.2 * x.0 - self.0 * x.2,
            self.0 * x.1 - self.1 * x.0,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    org: V,
    dir: V,
}
// assumes dir.len() == 1

pub type Color = V;

pub enum Reflection {
    Diffuse,    // 完全拡散面。いわゆるLambertian面。
    Specular,   // 理想的な鏡面。
    Refraction, // 理想的なガラス的物質。
}

pub struct Intersection {
    distance: f64,
    normal: V,
    position: V,
}

pub struct IntersectionWithID {
    intersection: Intersection,
    object_id: usize,
}

pub struct Sphere {
    radius: f64,
    center: V,
    emission: Color,
    color: Color,
    reflection: Reflection,
}

impl Sphere {
    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        use std::f64::EPSILON;
        // A ray intersects with a sphere iff
        // (self.center - (ray.org + ray.dir * t))^2 =< self.radius^2
        // which means checking whether the discriminant is positive
        // for the quadratic formular for t is sufficient.
        let p_o = self.center - ray.org;
        let b = p_o.dot(&ray.dir);
        let d4 = b * b - p_o.dot(&p_o) + self.radius * self.radius;

        if d4 < 0.0 {
            None
        } else {
            let d4_sq = d4.sqrt();
            let t1 = b - d4_sq;
            let t2 = b + d4_sq;
            if t1 < EPSILON && t2 < EPSILON {
                None
            } else {
                let distance = if t1 > EPSILON { t1 } else { t2 };
                let position = ray.org + ray.dir * distance;
                let normal = (position - self.center).normalize();
                Some(Intersection {
                    distance,
                    normal,
                    position,
                })
            }
        }
    }
}

pub static SPHERES: [Sphere; 10] = [
    Sphere {
        radius: 1e5,
        center: V(1e5 + 1.0, 40.8, 81.6),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.75, 0.25, 0.25),
        reflection: Reflection::Diffuse,
    }, // Left Wall
    Sphere {
        radius: 1e5,
        center: V(-1e5 + 99.0, 40.8, 81.6),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.25, 0.25, 0.75),
        reflection: Reflection::Diffuse,
    }, // Right Wall
    Sphere {
        radius: 1e5,
        center: V(50.0, 40.8, 1e5),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.75, 0.75, 0.75),
        reflection: Reflection::Diffuse,
    }, // Back Wall
    Sphere {
        radius: 1e5,
        center: V(50.0, 40.8, -1e5 + 250.0),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.0, 0.0, 0.0),
        reflection: Reflection::Diffuse,
    }, // Front Wall
    Sphere {
        radius: 1e5,
        center: V(50.0, 1e5, 81.6),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.75, 0.75, 0.75),
        reflection: Reflection::Diffuse,
    }, // Floor
    Sphere {
        radius: 1e5,
        center: V(50.0, -1e5 + 81.6, 81.6),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.75, 0.75, 0.75),
        reflection: Reflection::Diffuse,
    }, // Ceiling
    Sphere {
        radius: 20.0,
        center: V(65.0, 20.0, 20.0),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.25, 0.75, 0.25),
        reflection: Reflection::Diffuse,
    }, // Green Ball
    Sphere {
        radius: 16.5,
        center: V(27.0, 16.5, 47.0),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.99, 0.99, 0.99),
        reflection: Reflection::Specular,
    }, // Mirror
    Sphere {
        radius: 16.5,
        center: V(77.0, 16.5, 78.0),
        emission: V(0.0, 0.0, 0.0),
        color: V(0.99, 0.99, 0.99),
        reflection: Reflection::Refraction,
    }, // Glass
    Sphere {
        radius: 15.0,
        center: V(50.0, 90.0, 81.6),
        emission: V(36.0, 36.0, 36.0),
        color: V(0.0, 0.0, 0.0),
        reflection: Reflection::Diffuse,
    }, // Light
];

pub fn intersect_scene(ray: &Ray) -> Option<IntersectionWithID> {
    use std::cmp::Ordering::{Greater, Less};
    (0..10)
        .filter_map(|i| {
            SPHERES[i]
                .intersect(ray)
                .map(|intersection| IntersectionWithID {
                    intersection,
                    object_id: i,
                })
        })
        .min_by(|x, y| {
            // iffish floating point comparison
            // TODO: check if this actually *is* the correct thing to do
            if x.intersection.distance < y.intersection.distance {
                Less
            } else {
                Greater
            }
        })
}

const KDEPTH: i32 = 5;
const KDEPTH_LIMIT: i32 = 64;

pub fn radiance(ray: &Ray, seed: Seed, depth: i32) -> (Color, Seed) {
    use std::f64::consts::PI;
    use std::f64::EPSILON;
    match intersect_scene(ray) {
        None => (V(0.0, 0.0, 0.0), seed), // return background color
        Some(IntersectionWithID {
            intersection,
            object_id,
        }) => {
            let mut seed = seed;

            let current_object = &SPHERES[object_id];
            let orienting_normal = if intersection.normal.dot(&ray.dir) < 0.0 {
                intersection.normal
            } else {
                intersection.normal * -1.0
            };
            let mut russian_roulette_probability = current_object
                .color
                .0
                .max(current_object.color.1)
                .max(current_object.color.2)
                * if depth > KDEPTH_LIMIT {
                    0.5f64.powi(depth - KDEPTH_LIMIT)
                } else {
                    1.0
                };

            if depth > KDEPTH {
                let res = next01(seed);
                seed = res.1;
                if res.0 >= russian_roulette_probability {
                    return (current_object.emission, seed);
                }
            } else {
                russian_roulette_probability = 1.0;
            }

            let (incoming_radiance, weight) = match current_object.reflection {
                Reflection::Diffuse => {
                    let w = orienting_normal;
                    let u = if w.0.abs() > EPSILON {
                        V(0.0, 1.0, 0.0).cross(&w).normalize()
                    } else {
                        V(1.0, 0.0, 0.0).cross(&w).normalize()
                    };
                    let v = w.cross(&u);
                    let res = next01(seed);
                    seed = res.1;
                    let r1 = 2.0 * PI * res.0;
                    let res = next01(seed);
                    seed = res.1;
                    let r2 = res.0;
                    let r2_sq = r2.sqrt();
                    let dir = (u * r1.cos() * r2_sq + v * r1.sin() * r2_sq + w * (1.0 - r2).sqrt())
                        .normalize();
                    let res = radiance(
                        &Ray {
                            org: intersection.position,
                            dir,
                        },
                        seed,
                        depth + 1,
                    );
                    seed = res.1;
                    (res.0, current_object.color / russian_roulette_probability)
                }
                Reflection::Specular => {
                    let res = radiance(
                        &Ray {
                            org: intersection.position,
                            dir: ray.dir
                                - intersection.normal * 2.0 * intersection.normal.dot(&ray.dir),
                        },
                        seed,
                        depth + 1,
                    );
                    seed = res.1;
                    (res.0, current_object.color / russian_roulette_probability)
                }
                Reflection::Refraction => {
                    let reflection_ray = Ray {
                        org: intersection.position,
                        dir: ray.dir
                            - intersection.normal * 2.0 * intersection.normal.dot(&ray.dir),
                    };
                    let into = intersection.normal.dot(&orienting_normal) > 0.0;

                    let nc = 1.0;
                    let nt = 1.5;
                    let nnt = if into { nc / nt } else { nt / nc };
                    let ddn = ray.dir.dot(&orienting_normal);
                    let cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

                    if cos2t < 0.0 {
                        let res = radiance(&reflection_ray, seed, depth + 1);
                        seed = res.1;
                        (res.0, current_object.color / russian_roulette_probability)
                    } else {
                        let refraction_ray = Ray {
                            org: intersection.position,
                            dir: (ray.dir * nnt - intersection.normal * if into {
                                1.0
                            } else {
                                -1.0
                            }
                                * (ddn * nnt + cos2t.sqrt()))
                                .normalize(),
                        };

                        let a = nt - nc;
                        let b = nt + nc;
                        let r0 = (a * a) / (b * b);
                        let c = 1.0 - (if into {
                            -ddn
                        } else {
                            refraction_ray.dir.dot(&(orienting_normal * -1.0))
                        });
                        let re = r0 + (1.0 - r0) * c.powi(5);
                        let nnt2 = (if into { nc / nt } else { nt / nc }).powi(2);
                        let tr = (1.0 - re) * nnt2;

                        let probability = 0.25 + 0.5 * re;
                        if depth > 2 {
                            let res = next01(seed);
                            seed = res.1;
                            if res.0 < probability {
                                let res = radiance(&reflection_ray, seed, depth + 1);
                                seed = res.1;
                                (
                                    res.0 * re,
                                    current_object.color
                                        / (probability * russian_roulette_probability),
                                )
                            } else {
                                let res = radiance(&refraction_ray, seed, depth + 1);
                                seed = res.1;
                                (
                                    res.0 * tr,
                                    current_object.color
                                        / ((1.0 - probability) * russian_roulette_probability),
                                )
                            }
                        } else {
                            let res1 = radiance(&reflection_ray, seed, depth + 1);
                            seed = res1.1;
                            let res2 = radiance(&refraction_ray, seed, depth + 1);
                            seed = res2.1;
                            (
                                res1.0 * re + res2.0 * tr,
                                current_object.color / russian_roulette_probability,
                            )
                        }
                    }
                }
            };
            (current_object.emission + weight * incoming_radiance, seed)
        }
    }
}

pub fn render(width: usize, height: usize, samples: usize, supersamples: usize) {
    let camera_position = V(50.0, 52.0, 220.0);
    let camera_dir = V(0.0, -0.04, -1.0).normalize();
    let camera_up = V(0.0, 1.0, 0.0);

    let screen_width = 30.0 * width as f64 / height as f64;
    let screen_height = 30.0;
    let screen_dist = 40.0;

    let screen_x = camera_dir.cross(&camera_up).normalize() * screen_width;
    let screen_y = screen_x.cross(&camera_dir).normalize() * screen_height;
    let screen_center = camera_position + camera_dir * screen_dist;

    //let mut image = vec![V(0.0, 0.0, 0.0); width * height];
    println!(
        "{}x{} {} spp",
        width,
        height,
        samples * supersamples * supersamples
    );

    let image: Vec<_> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            println!("Rendering (y = {})", y);

            (0..width).into_par_iter().map(move |x| {
                let mut seed = init_xorshift((y * width + x) as u32 + 1);
                let mut ret = V(0.0, 0.0, 0.0);
                for sy in 0..supersamples {
                    for sx in 0..supersamples {
                        let mut accum = V(0.0, 0.0, 0.0);
                        for _ in 0..samples {
                            let rate = 1.0 / supersamples as f64;
                            let r1 = sx as f64 * rate + rate / 2.0;
                            let r2 = sy as f64 * rate + rate / 2.0;
                            let screen_position = screen_center
                                + screen_x * ((r1 + x as f64) / width as f64 - 0.5)
                                + screen_y * ((r2 + y as f64) / height as f64 - 0.5);
                            let dir = (screen_position - camera_position).normalize();

                            let res = radiance(
                                &Ray {
                                    org: camera_position,
                                    dir,
                                },
                                seed,
                                0,
                            );
                            seed = res.1;
                            accum = accum + res.0 / (samples * supersamples * supersamples) as f64;
                        }
                        ret = ret + accum;
                    }
                }
                ret
            })
        })
        .collect();

    save_ppm_file("image.ppm", &image, width, height);
}
