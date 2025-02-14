import numpy as np
import onnxruntime as ort

class VelocityDealiaser:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def dealias(self, velocity, nyquist):
        """
        velocity: [frames,az,rng,1]
        nyquist: [1]
        """
        n_frames = 1
        assert velocity.shape[0]==n_frames
        assert velocity.shape[1]==720 #number of azimuths
        velocity=velocity[None,...]
        nyquist=nyquist[None,...]
        # Handle 720 azimuth case
        recombine = False
        if velocity.shape[2]==720:
            s = velocity.shape
            velocity = np.reshape(velocity,(s[0],s[1],s[2]//2,2,s[3],s[4]))
            velocity = np.transpose(velocity,(0,3,1,2,4,5))
            velocity = np.reshape(velocity, (2*s[0],s[1],s[2]//2,s[3],s[4]))
            nyquist = np.stack((nyquist,nyquist))
            nyquist = np.transpose(nyquist,(1,0,2))
            nyquist = np.reshape(nyquist,(-1,n_frames,1))
            recombine = True

        # Pad edges
        pad_deg = 12
        velocity = np.concatenate((velocity[:,:,-pad_deg:,:,:],
                                 velocity,
                                 velocity[:,:,:pad_deg,:,:]), axis=2)

        velocity[velocity<=-64] = np.nan

        # Run inference
        input_data = {
            'vel': velocity.astype(np.float32),
            'nyq': nyquist.astype(np.float32)
        }
        output_data = self.session.run(None, input_data)
        dealiased_vel = output_data[1].copy()
        dealiased_vel = dealiased_vel[:, pad_deg:-pad_deg, : :]

        # Recombine if needed
        if recombine:
            s = dealiased_vel.shape
            dealiased_vel = np.reshape(dealiased_vel,(s[0]//2,2,s[1],s[2],s[3]))
            dealiased_vel = np.transpose(dealiased_vel,(0,2,1,3,4))
            dealiased_vel = np.reshape(dealiased_vel,(s[0]//2,-1,s[2],s[3]))
        
        dealiased_vel[dealiased_vel<=-64] = np.nan

        return dealiased_vel[0, :, :, 0]