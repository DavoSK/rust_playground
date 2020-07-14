use glow::*;
use glutin::event::{Event, WindowEvent};
use glutin::event_loop::ControlFlow;
use nalgebra_glm as na;

struct Mesh {
    vertex_array: u32
}

impl Mesh { 
    
    pub fn bind(&self, gl: &glow::Context) {
        //NOTE: bind vertex array
        unsafe { gl.bind_vertex_array(Some(self.vertex_array)); };
    } 

    pub fn render(&self, gl: &glow::Context, program: Program) {
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
            for i in 0..cubes.len()
            {
                self.bind(&gl);
                let model_mat: na::Mat4 = na::translate(&na::identity(), &cubes[i]);
                println!("model mat: {:?}", model_mat);

                let model_loc: Option<glow::UniformLocation> = gl.get_uniform_location(program, "model");
                gl.uniform_matrix_4_f32_slice(model_loc.as_ref(), false, na::value_ptr(&model_mat));
                gl.draw_arrays(glow::TRIANGLES, 0, 36);
            }
        };
    }
}

fn create_mesh(gl: &glow::Context) -> Mesh { 

    let vertex_array: u32;
    unsafe { 
        //NOTE: create vertex  array 
        vertex_array = gl
            .create_vertex_array()
            .expect("Cannot create vertex array");
        
        //NOTE: bind vertex array
        gl.bind_vertex_array(Some(vertex_array));

        //NOTE: fill vertices into create array
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

        let buffer_data: &[u8] = std::slice::from_raw_parts(vertices.as_ptr() as *const u8, std::mem::size_of::<f32>() * vertices.len());
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, buffer_data, glow::STATIC_DRAW);
    
        //NOTE: setup vertex attrib data
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, (3 * std::mem::size_of::<f32>()) as i32, 0);
        gl.enable_vertex_attrib_array(0);
    }

    return Mesh { vertex_array };
}

fn main() {
    let mut window_width: i32 = 1024;
    let mut window_height: i32 = 768;

    unsafe {
        let event_loop = glutin::event_loop::EventLoop::new();
        let shader_version = "#version 410";
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

        let testing_mesh: Mesh = create_mesh(&gl);

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

                    //NOTE: create static camera here
                    let proj_mat: na::Mat4 = na::perspective(window_width as f32 / window_height as f32, 45.0 * (na::pi::<f32>() / 180.0), 0.01, 500.0);
                    let view_mat: na::Mat4 = na::translate(&na::identity(), &na::vec3(0.0, 0.0, -5.0));

                    let view_loc: Option<glow::UniformLocation> = gl.get_uniform_location(program, "view");
                    let proj_loc: Option<glow::UniformLocation> = gl.get_uniform_location(program, "projection");
                    gl.uniform_matrix_4_f32_slice(proj_loc.as_ref(), false, na::value_ptr(&proj_mat));
                    gl.uniform_matrix_4_f32_slice(view_loc.as_ref(), false, na::value_ptr(&view_mat));

                    testing_mesh.render(&gl, program);
                    windowed_context.swap_buffers().unwrap();
                }
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(physical_size) => {
                        windowed_context.resize(*physical_size);
                        window_width = physical_size.width as i32;
                        window_height = physical_size.height as i32;
                        gl.viewport(0, 0, window_width, window_height);
                    }
                    WindowEvent::CloseRequested => {
                        gl.delete_program(program);
                        //gl.delete_vertex_array(vertex_array);
                        *control_flow = ControlFlow::Exit
                    }
                    _ => (),
                },
                _ => (),
            }
        });
        
    }
}
