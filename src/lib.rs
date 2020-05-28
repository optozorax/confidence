use std::convert::TryFrom;
use std::ops::Mul;

/// Value from `-1` to `1` inclusively
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(Option<f64>);

impl Confidence {
	pub fn none() -> Self {
		Confidence(None)
	}

	pub fn get(&self) -> Option<f64> {
		self.0
	}
}

impl Default for Confidence {
	fn default() -> Self {
		Self(Some(0.0))
	}
}

impl TryFrom<f64> for Confidence {
	type Error = f64;

	fn try_from(value: f64) -> Result<Self, Self::Error> {
		if -1.0 <= value && value <= 1.0 {
			Ok(Self(Some(value)))
		} else {
			Err(value)
		}
	}
}

impl From<Weight> for Confidence {
	fn from(src: Weight) -> Confidence {
		let src = match src.get() {
			Some(x) => x,
			None => return Confidence(None),
		};

		if src <= 1.0 {
			// Can use unsafe constructor because [0; 1] - 1 = [-1; 0]
			Confidence(Some(src - 1.0))
		} else {
			// Can use unsafe constructor because 1 - 1 / (1; inf] = 1 - [0; 1) = (0; 1]
			Confidence(Some(1.0 - 1.0 / src))
		}
	}
}

impl Mul for Confidence {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Confidence::from(Weight::from(self) * Weight::from(rhs))
	}
}

/// Value from `0` to `inf` inclusively
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Weight(Option<f64>);

impl Default for Weight {
	fn default() -> Self {
		Self(Some(1.0))
	}
}

impl Weight {
	pub fn none() -> Self {
		Weight(None)
	}

	pub fn get(&self) -> Option<f64> {
		self.0
	}
}

impl TryFrom<f64> for Weight {
	type Error = f64;

	fn try_from(value: f64) -> Result<Self, Self::Error> {
		if 0.0 <= value {
			Ok(Self(Some(value)))
		} else {
			Err(value)
		}
	}
}

impl From<Confidence> for Weight {
	fn from(src: Confidence) -> Weight {
		let src = match src.get() {
			Some(x) => x,
			None => return Weight(None),
		};

		if src <= 0.0 {
			// Can use unsafe constructor because [-1; 0] + 1 = [0; 1]
			Weight(Some(src + 1.0))
		} else {
			// Can use unsafe constructor because 1 / (1 - (0; 1]) = 1 / [0; 1) = (1; inf]
			Weight(Some(1.0/(1.0 - src)))
		}
	}
}

impl Mul for Weight {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		let lhs = match self.get() {
			Some(x) => x,
			None => return Weight(None),
		};
		let rhs = match rhs.get() {
			Some(x) => x,
			None => return Weight(None),
		};

		if lhs == 0.0 && rhs == f64::INFINITY || 
		   rhs == 0.0 && lhs == f64::INFINITY {
			// We think that this value is undefined
			Weight(None)
		} else {
			// Can use unsafe constructor because [0; inf] * [0; inf] = [0; inf] in other cases
			Weight(Some(lhs * rhs))
		}
	}
}

/// Value from `0` to `1` inclusively
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
	pub fn get(&self) -> f64 {
		self.0
	}
}

impl TryFrom<f64> for Probability {
	type Error = f64;

	fn try_from(value: f64) -> Result<Self, Self::Error> {
		if 0.0 <= value && value <= 1.0 {
			Ok(Self(value))
		} else {
			Err(value)
		}
	}
}

impl From<Confidence> for Probability {
	fn from(src: Confidence) -> Probability {
		match src.get() {
			Some(x) => Probability((x + 1.0) / 2.0),
			None => Probability(0.5),
		}
	}
}

impl Mul for Probability {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Probability(self.0 * rhs.0)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn basic_properties() {
		// Weight type tests
		let w_zero = Weight::try_from(0.0).unwrap();
		let w_half = Weight::try_from(0.5).unwrap();
		let w_one = Weight::try_from(1.0).unwrap();
		let w_two = Weight::try_from(2.0).unwrap();
		let w_inf = Weight::try_from(f64::INFINITY).unwrap();
		let w_none = Weight::none();
		let ws = [w_zero, w_half, w_one, w_two, w_inf, w_none];

		assert_eq!(w_one, w_half * w_two);
		assert_eq!(w_zero, w_zero * w_two);
		assert_eq!(w_inf, w_inf * w_two);
		assert_eq!(w_none, w_zero * w_inf);

		for &w in &ws {
			// Neutral element don't change multiplier
			assert_eq!(w * w_one, w);
			if w != w_inf && w != w_none {
				// Zero with anything valid is zero
				assert_eq!(w * w_zero, w_zero);
			}
			if w != w_zero && w != w_none {
				// Inf with anything valid is inf
				assert_eq!(w * w_inf, w_inf);
			}
			for &ww in &ws {
				for &www in &ws {
					// Order of operations has no effects
					assert_eq!((w * ww) * www, (w * www) * ww);
				}
			}
		}

		// Confidence type tests
		let c_minus_one = Confidence::try_from(-1.0).unwrap();
		let c_minus_half = Confidence::try_from(-0.5).unwrap();
		let c_zero = Confidence::try_from(0.0).unwrap();
		let c_plus_half = Confidence::try_from(0.5).unwrap();
		let c_plus_one = Confidence::try_from(1.0).unwrap();
		let c_none = Confidence::none();
		let cs = [c_minus_one, c_minus_half, c_zero, c_plus_half, c_plus_one, c_none];

		assert_eq!(c_zero, c_minus_half * c_plus_half);
		assert_eq!(c_minus_one, c_minus_one * c_plus_half);
		assert_eq!(c_plus_one, c_plus_one * c_plus_half);
		assert_eq!(c_none, c_minus_one * c_plus_one);

		for &c in &cs {
			// Neutral element don't change multiplier
			assert_eq!(c * c_zero, c);
			if c != c_plus_one && c != c_none {
				// Minus one with anything valid is minus one
				assert_eq!(c * c_minus_one, c_minus_one);
			}
			if c != c_minus_one && c != c_none {
				// Plus one with anything valid is plus one
				assert_eq!(c * c_plus_one, c_plus_one);
			}
			for &cc in &cs {
				for &ccc in &cs {
					// Order of operations has no effects
					assert_eq!((c * cc) * ccc, (c * ccc) * cc);
				}
			}
		}

		// Into tests
		assert_eq!(Confidence::from(w_zero), c_minus_one);
		assert_eq!(Confidence::from(w_half), c_minus_half);
		assert_eq!(Confidence::from(w_one), c_zero);
		assert_eq!(Confidence::from(w_two), c_plus_half);
		assert_eq!(Confidence::from(w_inf), c_plus_one);
		assert_eq!(Confidence::from(w_none), c_none);

		assert_eq!(w_zero, Weight::from(c_minus_one));
		assert_eq!(w_half, Weight::from(c_minus_half));
		assert_eq!(w_one, Weight::from(c_zero));
		assert_eq!(w_two, Weight::from(c_plus_half));
		assert_eq!(w_inf, Weight::from(c_plus_one));
		assert_eq!(w_none, Weight::from(c_none));

		assert_eq!(Probability::try_from(0.0).unwrap(), Probability::from(c_minus_one));
		assert_eq!(Probability::try_from(0.25).unwrap(), Probability::from(c_minus_half));
		assert_eq!(Probability::try_from(0.5).unwrap(), Probability::from(c_zero));
		assert_eq!(Probability::try_from(0.75).unwrap(), Probability::from(c_plus_half));
		assert_eq!(Probability::try_from(1.0).unwrap(), Probability::from(c_plus_one));
		assert_eq!(Probability::try_from(0.5).unwrap(), Probability::from(c_none));
	}

	#[test]
	fn integral_check() {
		/// Integrate by Gauss method of third order
		#[allow(clippy::many_single_char_names)] // Because this is math
		fn integrate<T, F>((a, b): (f64, f64), n: usize, f: F) -> T where
			F: Fn(f64) -> T,
			T: Default +  
				std::ops::Sub<T, Output = T> + 
				std::ops::Add<T, Output = T> + 
				std::ops::Div<T, Output = T> + 
				std::ops::Mul<T, Output = T> +
				std::ops::Sub<f64, Output = T> + 
				std::ops::Add<f64, Output = T> + 
				std::ops::Div<f64, Output = T> + 
				std::ops::Mul<f64, Output = T> +
		{
			assert!(a <= b);

			let x1 = -(3.0/5.0 as f64).sqrt();
			let x2 = 0.0;
			let x3 = -x1;

			let q1 = 5.0/9.0;
			let q2 = 8.0/9.0;
			let q3 = q1;

			let h = (b-a)/(n as f64+1.0);
			let h2 = h/2.0;

			let ah2 = a + h2;

			(0..=n)
				.map(|i| ah2 + h*i as f64) // in center of current segment
				.map(|xk|
					f(xk + x1 * h2) * q1 +
					f(xk + x2 * h2) * q2 + 
					f(xk + x3 * h2) * q3
				)
				.fold(None, |acc: Option<T>, x| match acc {
					None => Some(x),
					Some(acc) => Some(acc + x),
				}).unwrap() * h / 2.0
		}

		/// Check function properties by integral
		fn fitness<F: Fn(f64, f64) -> f64>(f: F, n: usize) -> f64 {
			let all = (-1.0, 1.0);   // All domain of definition
			let minus = (-1.0, 0.0); // Reducing part
			let plus = (0.0, 1.0);   // Increasing part

			// Distance between numbers
			fn dist(a: f64, b: f64) -> f64 {
				(a-b).abs()
			}

			// Conditional distance between numbers
			fn if_dist<F: Fn(f64, f64) -> bool>(a: f64, b: f64, comp: F) -> f64 {
				if comp(a, b) { 0.0 } else { dist(a, b) }
			}
			let mut sum = 0.0;

			
			sum += integrate(all, n, |x| 
				dist(f(x, 0.0), x) +	 // Neutral element
				dist(f(x, -1.0), -1.0) + // -1 always get -1 with other number
				dist(f(x, 1.0), 1.0)	 // -1 always get -1 with other number
			);

			// Positive increased
			sum += integrate(all, n, |x|
				integrate(plus, n, |plus|
					if_dist(x, f(x, plus), |a, b| a <= b)
				)
			);

			// Negative reduced
			sum += integrate(all, n, |x|
				integrate(minus, n, |minus|
					if_dist(x, f(x, minus), |a, b| a >= b)
				)
			);

			// Domain of definition remain itself
			sum += integrate(all, n, |x|
				integrate(minus, n, |y|
					if_dist(-1.0, f(x, y), |a, b| a <= b) +
					if_dist(f(x, y), 1.0, |a, b| a <= b)
				)
			);

			sum += integrate(all, n, |x|
				integrate(all, n, |y|
					integrate(all, n, |z| {
						let fzx = f(z, x);
						let fzy = f(z, y);

						// Order of operations has no effects
						dist(f(fzx, y), f(fzy, x)) + 

						// Function must be monotonous and increasing
						if x > y {
							if_dist(fzx, fzy, |a, b| a >= b)
						} else {
							if_dist(fzx, fzy, |a, b| a <= b)
						}
					})
				)
			);

			sum
		}

		assert!(fitness(|x, y| (Confidence::try_from(x).unwrap() * Confidence::try_from(y).unwrap()).get().unwrap(), 10).abs() < 1e-9);
	}
}
