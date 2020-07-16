use glow::*;
use glutin::event::{Event, WindowEvent};
use glutin::event_loop::ControlFlow;
use nalgebra_glm as na;

mod transform;
use transform::Transform as Transform;

use glutin::event::{
    ElementState, KeyboardInput, VirtualKeyCode,
};

struct Camera { 
    aspect: f32, 
    fov: f32,
    near: f32,
    far: f32,
    moving_up: bool,
    moving_left: bool,
    moving_down: bool,
    moving_right: bool,
    moving_forward: bool,
    moving_backward: bool,
    transform: Transform
}

impl Camera {
    pub fn new(aspect: f32, fov: f32, near: f32, far: f32) -> Self {
        return Camera { 
            aspect,
            fov,
            near, 
            far,
            moving_up: false,
            moving_left: false,
            moving_down: false,
            moving_right: false,
            moving_forward: false,
            moving_backward: false,
            transform: Transform::new() 
        };
    }

    pub fn render(&mut self, gl: &glow::Context, program: Program) {
        unsafe { 
            let proj_mat: na::Mat4 = na::perspective(self.aspect, self.fov, self.near, self.far);
            let view_loc = gl.get_uniform_location(program, "view");
            let proj_loc = gl.get_uniform_location(program, "projection");
            gl.uniform_matrix_4_f32_slice(proj_loc.as_ref(), false, na::value_ptr(&proj_mat));
            gl.uniform_matrix_4_f32_slice(view_loc.as_ref(), false, na::value_ptr(&self.transform().get_world()));
        }
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov;
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    pub fn set_near_far(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
    }

    pub fn transform(&mut self) -> &mut Transform {
        return &mut self.transform;
    }

    pub fn update(&mut self) {
        /*let f = {
            let f = self.direction;
            let len = f.0 * f.0 + f.1 * f.1 + f.2 * f.2;
            let len = len.sqrt();
            (f.0 / len, f.1 / len, f.2 / len)
        };

        let up = (0.0, 1.0, 0.0);

        let s = (f.1 * up.2 - f.2 * up.1,
                 f.2 * up.0 - f.0 * up.2,
                 f.0 * up.1 - f.1 * up.0);

        let s = {
            let len = s.0 * s.0 + s.1 * s.1 + s.2 * s.2;
            let len = len.sqrt();
            (s.0 / len, s.1 / len, s.2 / len)
        };

        let u = (s.1 * f.2 - s.2 * f.1,
                 s.2 * f.0 - s.0 * f.2,
                 s.0 * f.1 - s.1 * f.0);

        let current_pos = self.transform();
        if self.moving_up {
            self.position.0 += u.0 * 0.01;
            self.position.1 += u.1 * 0.01;
            self.position.2 += u.2 * 0.01;
        }

        if self.moving_left {
            self.position.0 -= s.0 * 0.01;
            self.position.1 -= s.1 * 0.01;
            self.position.2 -= s.2 * 0.01;
        }

        if self.moving_down {
            self.position.0 -= u.0 * 0.01;
            self.position.1 -= u.1 * 0.01;
            self.position.2 -= u.2 * 0.01;
        }

        if self.moving_right {
            self.position.0 += s.0 * 0.01;
            self.position.1 += s.1 * 0.01;
            self.position.2 += s.2 * 0.01;
        }

        if self.moving_forward {
            self.position.0 += f.0 * 0.01;
            self.position.1 += f.1 * 0.01;
            self.position.2 += f.2 * 0.01;
        }

        if self.moving_backward {
            self.position.0 -= f.0 * 0.01;
            self.position.1 -= f.1 * 0.01;
            self.position.2 -= f.2 * 0.01;
        }*/

        let mut ourVector = na::vec3(2, 0, 0);
        ourVector += na::vec3(5, 5, 6);
        println!("{:?}", ourVector);
    }

    pub fn process_input(&mut self, event: &glutin::event::WindowEvent) {
        let input = match *event {
            glutin::event::WindowEvent::KeyboardInput { input, .. } => input,
            _ => return,
        };
        let pressed = input.state == glutin::event::ElementState::Pressed;
        let key = match input.virtual_keycode {
            Some(key) => key,
            None => return,
        };
        match key {
            glutin::event::VirtualKeyCode::Up => self.moving_up = pressed,
            glutin::event::VirtualKeyCode::Down => self.moving_down = pressed,
            glutin::event::VirtualKeyCode::A => self.moving_left = pressed,
            glutin::event::VirtualKeyCode::D => self.moving_right = pressed,
            glutin::event::VirtualKeyCode::W => self.moving_forward = pressed,
            glutin::event::VirtualKeyCode::S => self.moving_backward = pressed,
            _ => (),
        };
    }
}

struct Mesh {
    vertex_buffer: glow::Buffer,
    transform: Transform
}

impl Mesh { 
    pub fn new(gl: &glow::Context) -> Self {
        return unsafe { 
            let vertices: [f32; 108] = [
                -0.5, -0.5, -0.5,  
                0.5, -0.5, -0.5, 
                0.5,  0.5, -0.5, 
                0.5,  0.5, -0.5,
                -0.5,  0.5, -0.5,  
                -0.5, -0.5, -0.5,
    
                -0.5, -0.5,  0.5, 
                0.5, -0.5,  0.5,  
                0.5,  0.5,  0.5, 
                0.5,  0.5,  0.5,
                -0.5,  0.5,  0.5,
                -0.5, -0.5,  0.5,
    
                -0.5,  0.5,  0.5,
                -0.5,  0.5, -0.5,
                -0.5, -0.5, -0.5,
                -0.5, -0.5, -0.5,
                -0.5, -0.5,  0.5,
                -0.5,  0.5,  0.5,
    
                0.5,  0.5,  0.5,
                0.5,  0.5, -0.5,
                0.5, -0.5, -0.5,
                0.5, -0.5, -0.5,
                0.5, -0.5,  0.5,
                0.5,  0.5,  0.5,
    
                -0.5, -0.5, -0.5,
                0.5, -0.5, -0.5,
                0.5, -0.5,  0.5,
                0.5, -0.5,  0.5,
                -0.5, -0.5,  0.5,
                -0.5, -0.5, -0.5,
    
                -0.5,  0.5, -0.5,
                0.5,  0.5, -0.5,
                0.5,  0.5,  0.5,
                0.5,  0.5,  0.5,
                -0.5,  0.5,  0.5,
                -0.5,  0.5, -0.5
            ];
    
            let buffer = gl.create_buffer().expect("failed to create buffer");
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
            {
                let buffer_data: &[u8] = std::slice::from_raw_parts(vertices.as_ptr() as *const u8, std::mem::size_of::<f32>() * vertices.len());
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, buffer_data, glow::STATIC_DRAW);
            }
    
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, (3 * std::mem::size_of::<f32>()) as i32, 0);
            gl.enable_vertex_attrib_array(0);
    
            Mesh { 
                vertex_buffer: buffer, 
                transform: Transform::new() 
            }
        }
    }

    pub fn transform(&mut self) -> &mut Transform { 
        return &mut self.transform;
    }

    pub fn render(&mut self, gl: &glow::Context, program: Program) {
        unsafe { 
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertex_buffer));
            {
                //NOTE: cache, transform + unifrom locations
                let model_loc = gl.get_uniform_location(program, "model");
                gl.uniform_matrix_4_f32_slice(model_loc.as_ref(), false, na::value_ptr(&self.transform().get_world()));
                gl.draw_arrays(glow::TRIANGLES, 0, 36);
            }
        };
    }
}

fn main() {
    let mut window_width: i32 = 1024;
    let mut window_height: i32 = 768;

    let cubes: [na::Vec3; 10] = [
        na::vec3( 0.0,  0.0,  0.0),
        na::vec3( 2.0,  5.0, -15.0),
        na::vec3(-1.5, -2.2, -2.5),
        na::vec3(-3.8, -2.0, -12.3),
        na::vec3( 2.4, -0.4, -3.5),
        na::vec3(-1.7,  3.0, -7.5),
        na::vec3( 1.3, -2.0, -2.5),
        na::vec3( 1.5,  2.0, -2.5),
        na::vec3( 1.5,  0.2, -1.5),
        na::vec3(-1.3,  1.0, -1.5)
    ];

    unsafe {
        let event_loop = glutin::event_loop::EventLoop::new();
        let shader_version = "#version 330 core";
        let window_builder = glutin::window::WindowBuilder::new()
            .with_title("Hello triangle!")
            .with_inner_size(glutin::dpi::LogicalSize::new(window_width, window_height));
        
        let windowed_context = glutin::ContextBuilder::new()
            .with_vsync(true)
            .build_windowed(window_builder, &event_loop)
            .unwrap();

        let windowed_context = windowed_context.make_current().unwrap();
        let gl = glow::Context::from_loader_function(|s| {
            windowed_context.get_proc_address(s) as *const _
        });

        let program = gl.create_program().expect("Cannot create program");
        let (vertex_shader_source, fragment_shader_source) = (
            r#"layout (location = 0) in vec3 aPos;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main() {
                gl_Position = projection * view * model * vec4(aPos, 1.0f);
            }"#,
            r#"precision mediump float;
            out vec4 vColor;
            void main() {
                vColor = vec4(1.0, 1.0, 1.0, 1.0);
            }"#,
        );

        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_shader_source),
            (glow::FRAGMENT_SHADER, fragment_shader_source),
        ];

        let mut shaders = Vec::with_capacity(shader_sources.len());
        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl
                .create_shader(*shader_type)
                .expect("Cannot create shader");
            gl.shader_source(shader, &format!("{}\n{}", shader_version, shader_source));
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!(gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!(gl.get_program_info_log(program));
        }

        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        gl.clear_color(0.1, 0.2, 0.3, 1.0);
        gl.enable(glow::DEPTH_TEST);

        let mut camera: Camera = Camera::new(window_width as f32 / window_height as f32, 45.0 * (na::pi::<f32>() / 180.0), 0.01, 500.0);
        camera.transform().set_pos(na::vec3(0.0, 0.0, -5.0));

        let mut meshes: Vec<Mesh> = Vec::new();
        for pos in cubes.iter() {
            let mut testing_mesh: Mesh = Mesh::new(&gl);
            testing_mesh.transform().set_pos(*pos);
            meshes.push(testing_mesh);
        }

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::LoopDestroyed => {
                    return;
                }
                Event::MainEventsCleared => {
                    windowed_context.window().request_redraw();
                }
                Event::RedrawRequested(_) => {
                    gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                    gl.use_program(Some(program));

                    camera.update();
                    camera.render(&gl, program);

                    for mesh in &mut meshes {
                        mesh.render(&gl, program);
                    }

                    windowed_context.swap_buffers().unwrap();
                }
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(physical_size) => {
                        windowed_context.resize(*physical_size);
                        window_width = physical_size.width as i32;
                        window_height = physical_size.height as i32;
                        gl.viewport(0, 0, window_width, window_height);
                        camera.set_aspect(window_width as f32 / window_height as f32);
                    }

                    WindowEvent::CloseRequested => {
                        gl.delete_program(program);
                        *control_flow = ControlFlow::Exit
                    }
                    ev => camera.process_input(&ev),
                },
                _ => (),
            }
        });
        
    }
}
