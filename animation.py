from manim import *

class ContinuousRod(Scene):
    def construct(self):
        # 1. The continuous rod and PDE
        title = Tex("1. The Continuous Rod and PDE").to_edge(UP)
        self.play(Write(title))
        self.wait(2)

        # Draw the 1D Rod
        rod = Line(LEFT * 4, RIGHT * 4, stroke_width=25).shift(UP * 1)
        rod.set_color([BLUE, RED, RED, BLUE]) 
        
        # Labels for x=0 and x=1
        label_0 = MathTex("x=0").next_to(rod, DOWN).align_to(rod, LEFT)
        label_1 = MathTex("x=1").next_to(rod, DOWN).align_to(rod, RIGHT)
        
        rod_group = VGroup(rod, label_0, label_1)

        # Display Heat Equation
        heat_eq = MathTex(
            r"\frac{\partial u}{\partial t} = \alpha^2\frac{\partial ^2 u}{\partial x^2}"
        ).scale(1.5).next_to(rod_group, DOWN, buff=1)

        # Show rod and equation together
        self.play(Create(rod), Write(label_0), Write(label_1))
        self.play(Write(heat_eq))
        self.wait(3)

        # Transition: Fade out the rod and move the equation up to make room for components
        self.play(
            FadeOut(rod_group),
            heat_eq.animate.shift(UP * 2.5)
        )
        self.wait(1)

        # Display Definitions
        definitions = MathTex(
            r"u(t,x) &:= \text{Temperature} \\",
            r"t &:= \text{Time} \\",
            r"x &:= \text{Position on the rod}"
        ).next_to(heat_eq, DOWN, buff=1)

        self.play(FadeIn(definitions, shift=UP))
        self.wait(4)
        
        self.play(FadeOut(heat_eq), FadeOut(definitions), FadeOut(title))

class Discretisation(Scene):
    def construct(self):
        # 2. Discretisation
        title = Tex("2. Discretisation").to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Boundary Conditions
        bc_title = Tex("Boundary Conditions").scale(0.8).next_to(title, DOWN, buff=0.5)
        bc_eq = MathTex(r"u(t, 0) = 0 \quad \text{and} \quad u(t, 1) = 0")
        
        self.play(Write(bc_title))
        self.play(Write(bc_eq))
        self.wait(3)

        self.play(FadeOut(bc_title), FadeOut(bc_eq))

        # Initial Conditions
        ic_title = Tex("Initial Conditions").scale(0.8).next_to(title, DOWN, buff=0.5)
        ic_eq = MathTex(
            r"u(0, x) =",
            r"\begin{cases}",
            r"0 & \text{if } x \in [0, \frac{1}{3}) \cup (\frac{2}{3}, 1], \\",
            r"100 & \text{if } x \in [\frac{1}{3}, \frac{2}{3}]",
            r"\end{cases}"
        )

        self.play(Write(ic_title))
        self.play(Write(ic_eq))
        self.wait(4)
        
        self.play(FadeOut(ic_title), FadeOut(ic_eq))

        # --- NEW: Grid Visualization ---
        grid_title = Tex("Discretizing Space and Time").scale(0.8).next_to(title, DOWN, buff=0.5)
        self.play(Write(grid_title))

        # Create Axes for Space (x) and Time (t)
        axes = Axes(
            x_range=[0, 5, 1], 
            y_range=[0, 4, 1], 
            x_length=6, 
            y_length=4,
            axis_config={"include_tip": True}
        ).shift(DOWN * 0.5)
        
        x_label = axes.get_x_axis_label("x")
        t_label = axes.get_y_axis_label("t")

        # Create grid dots
        dots = VGroup()
        for x in range(0, 6):
            for y in range(0, 5):
                dots.add(Dot(axes.c2p(x, y), color=YELLOW, radius=0.06))

        self.play(Create(axes), Write(x_label), Write(t_label))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.02))
        self.wait(3)

        self.play(FadeOut(axes), FadeOut(x_label), FadeOut(t_label), FadeOut(dots), FadeOut(grid_title))
        # -------------------------------

        # Finite Differences & Taylor's Theorem
        fd_title = Tex("Approximating Derivatives (Finite Differences)").scale(0.8).next_to(title, DOWN, buff=0.5)
        self.play(Write(fd_title))

        taylor1 = MathTex(
            r"f(x+\Delta x) = f(x) + \Delta xf'(x) +\frac{\Delta x^2}{2!}f''(x) + \frac{\Delta x^3}{3!}f^{(3)}(x) + ..."
        ).scale(0.8)
        
        self.play(Write(taylor1))
        self.wait(2)

        # Rearranging step by step
        taylor_rearrange1 = MathTex(
            r"f(x+\Delta x) - f(x) = \Delta xf'(x) +\frac{\Delta x^2}{2!}f''(x) + \frac{\Delta x^3}{3!}f^{(3)}(x) + ..."
        ).scale(0.8)

        taylor_rearrange2 = MathTex(
            r"\frac{f(x+\Delta x) - f(x)}{\Delta x} = f'(x) +\frac{\Delta x}{2!}f''(x) + \frac{\Delta x^2}{3!}f^{(3)}(x) + ..."
        ).scale(0.8)

        self.play(TransformMatchingTex(taylor1, taylor_rearrange1))
        self.wait(2)
        self.play(TransformMatchingTex(taylor_rearrange1, taylor_rearrange2))
        self.wait(2)

        # Fun remark (Limit definition)
        limit_remark = MathTex(
            r"\lim_{\Delta x \rightarrow 0} \frac{f(x+\Delta x) - f(x)}{\Delta x} = f'(x)"
        ).scale(0.6).set_color(YELLOW).to_corner(DR)
        
        self.play(FadeIn(limit_remark, shift=LEFT))
        self.wait(2)

        # Big O Notation transformation
        big_o_1 = MathTex(
            r"\frac{f(x+\Delta x) - f(x)}{\Delta x} = f'(x) + \mathcal{O}(\Delta x)"
        ).scale(0.9)

        self.play(Transform(taylor_rearrange2, big_o_1))
        self.wait(3)

        self.play(FadeOut(taylor_rearrange2), FadeOut(limit_remark))

        # Second Derivatives
        taylor_minus = MathTex(
            r"f(x-\Delta x) = f(x) - \Delta x f'(x) + \frac{\Delta x^2}{2!}f''(x) - \frac{\Delta x ^3}{3!}f^{(3)}(x) + ..."
        ).scale(0.8)

        self.play(Write(taylor_minus))
        self.wait(2)

        taylor_sum = MathTex(
            r"f(x+\Delta x) + f(x-\Delta x) = 2f(x) + \frac{2\Delta x^2}{2!}f''(x) + \frac{2\Delta x^4}{4!}f^{(4)}(x) + ..."
        ).scale(0.8)

        self.play(TransformMatchingTex(taylor_minus, taylor_sum))
        self.wait(2)

        big_o_2 = MathTex(
            r"\frac{f(x+\Delta x) - 2f(x) + f(x-\Delta x)}{\Delta x^2} = f''(x) + \mathcal{O}(\Delta x^2)"
        ).scale(0.9)

        self.play(TransformMatchingTex(taylor_sum, big_o_2))
        self.wait(3)

        self.play(FadeOut(big_o_2), FadeOut(fd_title), FadeOut(title))

class HeatFlow(Scene):
    def construct(self):
        # 3. Heat Flow
        title = Tex("3. Heat Flow").to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Display discretizations side by side or top/bottom
        time_deriv = MathTex(
            r"\frac{u(t + \Delta t, x) - u(t, x)}{\Delta t} = \frac{\partial u}{\partial t} + \mathcal{O}(\Delta t)"
        ).scale(0.8).shift(UP*1.5)

        space_deriv = MathTex(
            r"\frac{u(t, x + \Delta x) - 2u(t,x) + u(t, x - \Delta x)}{\Delta x^2} = \frac{\partial ^2 u}{\partial x^2} + \mathcal{O}(\Delta x^2)"
        ).scale(0.8).next_to(time_deriv, DOWN, buff=1)

        self.play(Write(time_deriv))
        self.wait(2)
        self.play(Write(space_deriv))
        self.wait(3)

        self.play(FadeOut(time_deriv), FadeOut(space_deriv))

        # Fully discrete heat equation
        discrete_heat = MathTex(
            r"\frac{u(t + \Delta t, x) - u(t, x)}{\Delta t} = \alpha^2 \left(\frac{u(t, x + \Delta x) - 2u(t,x) + u(t, x - \Delta x)}{\Delta x^2}\right)"
        ).scale(0.8)

        self.play(Write(discrete_heat))
        self.wait(3)

        # Final rearranged equation
        final_eq = MathTex(
            r"u(t+\Delta t, x) = u(t,x) + \alpha^2\left(\frac{\Delta t}{\Delta x^2}\right) \Big(u(t, x+\Delta x) - 2u(t, x) + u(t, x- \Delta x)\Big)"
        ).scale(0.8).set_color(GREEN_C)

        self.play(TransformMatchingTex(discrete_heat, final_eq))
        self.wait(4)

        self.play(FadeOut(final_eq), FadeOut(title))