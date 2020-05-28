#![feature(proc_macro_hygiene)]

use std::convert::TryFrom;
use confidence::Confidence;
use inline_python::python;

fn main() {
	let n = 100;
	let iterator = (-n..=n).map(|x| x as f64 / n as f64);
	let x = iterator.clone().collect::<Vec<f64>>();
	let y = iterator.collect::<Vec<f64>>(); 
	let mut z = vec![vec![0.0; x.len()]; y.len()]; 
	for (ix, x) in x.iter().enumerate() {
		for (iy, y) in y.iter().enumerate() {
			z[ix][iy] = (Confidence::try_from(*x).unwrap() * Confidence::try_from(*y).unwrap()).get().unwrap_or(0.0);
		}
	}

	python! {
		// ugly hack
		import sys
		sys.argv = ["./myprog"]

		from mpl_toolkits.mplot3d import Axes3D

		import matplotlib.pyplot as plt
		from matplotlib import cm
		from matplotlib.ticker import LinearLocator, FormatStrFormatter
		import numpy as np

		fig = plt.figure()
		ax = fig.gca(projection="3d")

		X = np.array('x)
		Y = np.array('y)
		X, Y = np.meshgrid(X, Y)
		Z = np.array('z)

		surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
		                       linewidth=0, antialiased=False)

		ax.set_zlim(-1.01, 1.01)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter("%.01f"))

		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.show()
	}
}