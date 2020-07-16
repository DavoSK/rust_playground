use glow::*;
use nalgebra_glm as glm;
mod transform;
use transform::Transform as Transform;

struct Camera { 
    //NOTE: basic params
    aspect: f32, 
    fov: f32,
    near: f32,
    far: f32,
    //NOTE: controls
    moving_up: bool,
    moving_left: bool,
    moving_down: bool,
    moving_right: bool,
    moving_forward: bool,
    moving_backward: bool,
    //NOTE: positioning
    view: glm::Mat4,
    projection: glm::Mat4,
    position: glm::Vec3,
    forward: glm::Vec3,
    right: glm::Vec3,
    angles: glm::Vec2
}

pub fn vec3_mul(lhs: glm::Vec3, rhs: glm::Vec3) -> glm::Vec3 {
    return glm::Vec3::new(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
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
            view: glm::identity(),
            projection: glm::perspective(aspect, fov, near, far),
            position: glm::Vec3::new(0.0, 0.0, 0.0),
            forward: glm::Vec3::new(0.0, 0.0, 0.0),
            right: glm::Vec3::new(0.0, 0.0, 0.0),
            angles: glm::Vec2::new(0.0, 0.0)
        };
    }

    pub fn update_projection(&mut self) {
        self.projection = glm::perspective(self.aspect, self.fov, self.near, self.far);
    }

    pub fn set_pos(&mut self, pos: glm::Vec3) {
        self.position = pos;
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov;
        self.update_projection();
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.update_projection();
    }

    pub fn set_near_far(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
        self.update_projection();
    }

    pub fn render(&mut self, gl: &glow::Context, program: Program) {
        unsafe { 
            let view_loc = gl.get_uniform_location(program, "view");
            let proj_loc = gl.get_uniform_location(program, "projection");
            gl.uniform_matrix_4_f32_slice(proj_loc.as_ref(), false, glm::value_ptr(&self.projection));
            gl.uniform_matrix_4_f32_slice(view_loc.as_ref(), false, glm::value_ptr(&self.view));
        }
    }

    pub fn update(&mut self, delta_time: f32) {
        let scalar_speed: f32 = 32.0 * delta_time;
        let speed = glm::Vec3::new(scalar_speed, scalar_speed, scalar_speed);

        if self.moving_forward {
            self.position += vec3_mul(self.forward, speed);
        }

        if self.moving_left {
            self.position += vec3_mul(self.right, speed * -1.0);
        }

        if self.moving_right {
            self.position += vec3_mul(self.right, speed);
        }

        if self.moving_backward {
            self.position += vec3_mul(self.forward, speed * -1.0);
        }

        let rot_y = glm::rotate_y(&glm::identity(), self.angles.x);
        let rot_x = glm::rotate_x(&glm::identity(), -self.angles.y);

        let translation = glm::translate(&glm::identity(), &vec3_mul(self.position, glm::Vec3::new(-1.0, -1.0, -1.0)));
        let rotation = rot_y * rot_x;
        let camera = translation * rotation;

        self.view = glm::inverse(&camera);
        self.forward = glm::Vec3::new(camera[8], camera[9], camera[10]);
        self.right = glm::cross(&self.forward, &glm::Vec3::new(0.0, 1.0, 0.0));        
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
                //TODO: cache, transform + uniform locations
                let model_loc = gl.get_uniform_location(program, "model");
                gl.uniform_matrix_4_f32_slice(model_loc.as_ref(), false, glm::value_ptr(&self.transform.get_world()));
                gl.draw_arrays(glow::TRIANGLES, 0, 36);
            }
        };
    }
}

pub enum Action {
    Stop,
    Continue,
}

pub fn start_loop<F>(event_loop: glutin::event_loop::EventLoop<()>, mut callback: F)->! where F: 'static + FnMut(&Vec<glutin::event::Event<()>>) -> Action {
    let mut events_buffer = Vec::new();
    let mut next_frame_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        let run_callback = match event.to_static() {
            Some(glutin::event::Event::NewEvents(cause)) => {
                match cause {
                    glutin::event::StartCause::ResumeTimeReached { .. } | glutin::event::StartCause::Init => {
                        true
                    },
                    _ => false
                }
            },
            Some(event) => {
                events_buffer.push(event);
                false
            }
            None => {
                false
            },
        };

        let action = if run_callback {
            let action = callback(&events_buffer);
            next_frame_time = std::time::Instant::now() + std::time::Duration::from_millis(5);
            events_buffer.clear();
            action
        } else {
            Action::Continue
        };

        match action {
            Action::Continue => {
                *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
            },
            Action::Stop => *control_flow = glutin::event_loop::ControlFlow::Exit
        }
    })
}


fn main() {
    let mut window_width: i32 = 1024;
    let mut window_height: i32 = 768;

    let cubes: [glm::Vec3; 10] = [
        glm::Vec3::new( 0.0,  0.0,  0.0),
        glm::Vec3::new( 2.0,  5.0, -15.0),
        glm::Vec3::new(-1.5, -2.2, -2.5),
        glm::Vec3::new(-3.8, -2.0, -12.3),
        glm::Vec3::new( 2.4, -0.4, -3.5),
        glm::Vec3::new(-1.7,  3.0, -7.5),
        glm::Vec3::new( 1.3, -2.0, -2.5),
        glm::Vec3::new( 1.5,  2.0, -2.5),
        glm::Vec3::new( 1.5,  0.2, -1.5),
        glm::Vec3::new(-1.3,  1.0, -1.5)
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

        let mut camera: Camera = Camera::new(window_width as f32 / window_height as f32, 45.0 * (glm::pi::<f32>() / 180.0), 0.01, 500.0);
        camera.set_pos(glm::Vec3::new(4.0, 2.0, -8.0));

        let mut meshes: Vec<Mesh> = Vec::new();
        for pos in cubes.iter() {
            let mut testing_mesh: Mesh = Mesh::new(&gl);
            testing_mesh.transform().set_pos(*pos);
            meshes.push(testing_mesh);
        }

        //TODO: split update & rendering
        //we will call update max 30fps locked
        let mut last_time = std::time::Instant::now();
        start_loop(event_loop, move |events| { 
            let delta_time = (std::time::Instant::now() - last_time).as_secs_f32();
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            gl.use_program(Some(program));

            camera.update(delta_time);
            camera.render(&gl, program);

            for mesh in &mut meshes {
                mesh.render(&gl, program);
            }

            windowed_context.swap_buffers().unwrap();
            last_time = std::time::Instant::now();

            let mut action = Action::Continue;
            for event in events {
                match event {
                    glutin::event::Event::WindowEvent { event, .. } => match event {
                        glutin::event::WindowEvent::CloseRequested => action = Action::Stop,
                        glutin::event::WindowEvent::Resized(size) => {
                            window_width = size.width as i32;
                            window_height =  size.height as i32;
                            gl.viewport(0, 0, window_width, window_height);
                            camera.set_aspect(window_width as f32 / window_height as f32);
                        },
                        ev => camera.process_input(&ev),
                    },
                    _ => (),
                }
            };
            action
        });        
    }
}
