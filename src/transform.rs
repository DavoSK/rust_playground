use nalgebra_glm as glm;

pub struct Transform {  
    translation: glm::Mat4,
    rotation: glm::Mat4, 
    scale: glm::Mat4,
}

impl Transform {
    pub fn new() -> Self { 
        return Transform { translation: glm::identity(), rotation: glm::identity(), scale: glm::identity() }; 
    }

    pub fn set_translation(&mut self, translation: glm::Mat4) -> &mut Self {
        self.translation = translation;
        return self;
    } 

    pub fn set_pos(&mut self, pos: glm::Vec3) -> &mut Self {
        self.translation = glm::translate(&glm::identity(), &pos);
        return self;
    }

    pub fn set_rot(&mut self, rot: glm::Quat) -> &mut Self{
        self.rotation = glm::quat_to_mat4(&rot);
        return self;
    }

    pub fn set_rot_mat(&mut self, rot: glm::Mat4) -> &mut Self {
        self.rotation = rot;
        return self;
    }
    
    pub fn set_scale(&mut self, scale: glm::Vec3) -> &mut Self {
        self.scale = glm::scale(&glm::identity(), &scale);
        return self;
    }

    pub fn get_world(&self) -> glm::Mat4 {
        return self.translation * self.rotation * self.scale;
    }
}