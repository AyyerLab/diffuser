extern "C" {

__global__
void gen_dens(const float *positions, const float *atom_f0, const long long num_atoms,
              const long long* size, float *dens) {
    int n = blockIdx.x * blockDim.x + threadIdx.x ;
    if (n >= num_atoms)
        return ;

    float tx = positions[n*3 + 0] ;
    float ty = positions[n*3 + 1] ;
    float tz = positions[n*3 + 2] ;
    float val = atom_f0[n] ;
    int ix = __float2int_rd(tx), iy = __float2int_rd(ty), iz = __float2int_rd(tz) ;
    if ((ix < 0) || (ix > size[0] - 2) ||
        (iy < 0) || (iy > size[1] - 2) ||
        (iz < 0) || (iz > size[2] - 2))
        return ;
    float fx = tx - ix, fy = ty - iy, fz = tz - iz ;
    float cx = 1. - fx, cy = 1. - fy, cz = 1. - fz ;

    atomicAdd(&dens[ix*size[1]*size[2] + iy*size[2]+ iz], val*cx*cy*cz) ;
    atomicAdd(&dens[ix*size[1]*size[2] + iy*size[2]+ iz+1], val*cx*cy*fz) ;
    atomicAdd(&dens[ix*size[1]*size[2] + (iy+1)*size[2]+ iz], val*cx*fy*cz) ;
    atomicAdd(&dens[ix*size[1]*size[2] + (iy+1)*size[2]+ iz+1], val*cx*fy*fz) ;
    atomicAdd(&dens[(ix+1)*size[1]*size[2] + iy*size[2]+ iz], val*fx*cy*cz) ;
    atomicAdd(&dens[(ix+1)*size[1]*size[2] + iy*size[2]+ iz+1], val*fx*cy*fz) ;
    atomicAdd(&dens[(ix+1)*size[1]*size[2] + (iy+1)*size[2]+ iz], val*fx*fy*cz) ;
    atomicAdd(&dens[(ix+1)*size[1]*size[2] + (iy+1)*size[2]+ iz+1], val*fx*fy*fz) ;
}

__global__
void get_qrad(const long long *sizes, const double *qvox, double *qrad) {
	int x = blockIdx.x * blockDim.x + threadIdx.x ;
	int y = blockIdx.y * blockDim.y + threadIdx.y ;
	int z = blockIdx.z * blockDim.z + threadIdx.z ;

	int tx = x - sizes[0] / 2 ;
	int ty = y - sizes[1] / 2 ;
	int tz = z - sizes[2] / 2 ;

	double qx = tx * qvox[0] + ty * qvox[1] + tz * qvox[2] ;
	double qy = tx * qvox[3] + ty * qvox[4] + tz * qvox[5] ;
	double qz = tx * qvox[6] + ty * qvox[7] + tz * qvox[8] ;

	qrad[x*sizes[1]*sizes[2] + y*sizes[2] + z] = sqrt(qx*qx + qy*qy + qz*qz) ;
}

} // extern C
