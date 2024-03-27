from manimlib import *

class ComplexTransformation(Scene):
    def construct(self):
        # Create a complex plane
        plane = ComplexPlane()
        plane.prepare_for_nonlinear_transform()
        self.add(plane)
        
        # Define the transformation function
        def transformation(z):
            return 0.3*z + 0.3*np.sin(z)

        # Animate the transformation of the complex plane
        transformed_plane = plane.copy()
        transformed_plane.apply_complex_function(transformation)
        self.play(Transform(plane, transformed_plane), run_time=4)

        self.wait()
