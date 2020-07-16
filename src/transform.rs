
use nalgebra_glm as na;

pub struct Transform {  
    translation: na::Mat4,
    rotation: na::Mat4, 
    scale: na::Mat4,
}

impl Transform {
    pub fn new() -> Self { 
        return Transform { translation: na::identity(), rotation: na::identity(), scale: na::identity() }; 
    }

    /*pub fn get_pos(&mut self) -> &mut na::Vec3 {
        return &mut na::translation(&self.translation);
    }*/

    pub fn set_pos(&mut self, pos: na::Vec3) -> &mut Self {
        self.translation = na::translate(&na::identity(), &pos);
        return self;
    }

    pub fn set_rot(&mut self, rot: na::Quat) -> &mut Self{
        self.rotation = na::quat_to_mat4(&rot);
        return self;
    }

    pub fn set_scale(&mut self, scale: na::Vec3) -> &mut Self {
        self.scale = na::scale(&na::identity(), &scale);
        return self;
    }

    pub fn get_world(&self) -> na::Mat4 {
        return self.translation * self.rotation * self.scale;
    }
}