/* OpenCL ray tracing tutorial by Sam Lapere, 2016
http://raytracey.blogspot.com */

struct Ray{
	float3 origin;
	float3 dir;
};

struct Sphere{
	float radius;
	float3 pos;
	float3 emi;
	float3 color;
};

struct Triangle{
	float3 v0;
	float3 v1;
	float3 v2;
	float3 color;
	float3 center;
};

bool intersect_triangle(const struct Triangle* triangle, const struct Ray* ray, float* t)
{
	float3 v0v1 = triangle->v1 - triangle->v0;
	float3 v0v2 = triangle->v2 - triangle->v0;
	float3 pvec = cross(ray->dir, v0v2); 

	float det = dot(v0v1, pvec);
	if(det<-1e-8f)
	{
		return false;
	}
	if(fabs(det) < -1e-8f)
	{
		return false;
	}
	float invDet = 1.0f/det;
	float3 tvec = ray->origin - triangle->v0;

	float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1)
    {
        return false;
    }
	float3 qvec = cross(tvec, v0v1);
	float v = dot(ray->dir, qvec)*invDet;
	if (v < 0 || u + v > 1)
    {
        return false;
    }

	*t = dot(v0v2, qvec) * invDet;
	return true;
}


bool intersect_sphere(const struct Sphere* sphere, const struct Ray* ray, float* t)
{
	float3 rayToCenter = sphere->pos - ray->origin;

	/* calculate coefficients a, b, c from quadratic equation */

	/* float a = dot(ray->dir, ray->dir); // ray direction is normalised, dotproduct simplifies to 1 */ 
	float b = dot(rayToCenter, ray->dir);
	float c = dot(rayToCenter, rayToCenter) - sphere->radius*sphere->radius;
	float disc = b * b - c; /* discriminant of quadratic formula */

	/* solve for t (distance to hitpoint along ray) */

	if (disc < 0.0f) return false;
	else *t = b - sqrt(disc);

	if (*t < 0.0f){
		*t = b + sqrt(disc);
		if (*t < 0.0f) return false; 
	}

	else return true;
}

struct Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height){

	float fx = (float)x_coord / (float)width;  /* convert int in range [0 - width] to float in range [0-1] */
	float fy = (float)y_coord / (float)height; /* convert int in range [0 - height] to float in range [0-1] */

	/* calculate aspect ratio */
	float aspect_ratio = (float)(width) / (float)(height);
	float fx2 = (fx - 0.5f) * aspect_ratio;
	float fy2 = fy - 0.5f;

	/* determine position of pixel on screen */
	float3 pixel_pos = (float3)(fx2, -fy2, 0.0f);

	/* create camera ray*/
	struct Ray ray;
	ray.origin = (float3)(0.0f, 0.0f, 10.0f); /* fixed camera position */
	ray.dir = normalize(pixel_pos - ray.origin); /* ray direction is vector from camera to pixel */

	return ray;
}

__kernel void render_kernel(__global float3* output, int width, int height, int rendermode)
{
	const int work_item_id = get_global_id(0);		/* the unique global id of the work item for the current pixel */
	int x_coord = work_item_id % width;					/* x-coordinate of the pixel */
	int y_coord = work_item_id / width;					/* y-coordinate of the pixel */

	float fx = (float)x_coord / (float)width;  /* convert int in range [0 - width] to float in range [0-1] */
	float fy = (float)y_coord / (float)height; /* convert int in range [0 - height] to float in range [0-1] */

	/*create a camera ray */
	struct Ray camray = createCamRay(x_coord, y_coord, width, height);

	/* create and initialise a sphere */
	struct Sphere sphere1;
	sphere1.radius = 0.4f;
	sphere1.pos = (float3)(0.0f, 0.0f, 3.0f);
	sphere1.color = (float3)(0.9f, 0.3f, 0.0f);


	/* create and initialise a triangle
	struct Triangle sphere1;
	sphere1.v0 = (float3)(0.0f, 0.0f, 0.0f);
	sphere1.v1 = (float3)(0.f, 1.f, 0.f);
	sphere1.v2 = (float3)(1.0f, 0.f, 0.0f);
	sphere1.center = (float3)(1.0f, 0.75f, 2.0f);
	sphere1.color = (float3)(0.9f, 0.3f, 0.0f); */


	/* intersect ray with sphere */
	float t = 1e20;
	intersect_sphere(&sphere1, &camray, &t);
	/*intersect_triangle(&sphere1, &camray, &t);*/

	/* if ray misses sphere, return background colour 
	background colour is a blue-ish gradient dependent on image height */
	if (t > 1e19 && rendermode != 1)
	{ 
		output[work_item_id] = (float3)(fy * 0.1f, fy * 0.3f, 0.3f);
		return;
	}


	/* for more interesting lighting: compute normal 
	and cosine of angle between normal and ray direction */
	float3 hitpoint = camray.origin + camray.dir * t;
	/*float3 normal = normalize(hitpoint - sphere1.center);*/
	float3 normal = normalize(hitpoint - sphere1.pos);
	float cosine_factor = dot(normal, camray.dir) * -1.0f;
	
	output[work_item_id] = sphere1.color * cosine_factor;

	/* six different rendermodes */
	if (rendermode == 1) output[work_item_id] = (float3)(fx, fy, 0); /* simple interpolated colour gradient based on pixel coordinates */
	else if (rendermode == 2) output[work_item_id] = sphere1.color;  /* raytraced sphere with plain colour */
	else if (rendermode == 3) output[work_item_id] = sphere1.color * cosine_factor; /* with cosine weighted colour */
	else if (rendermode == 4) output[work_item_id] = sphere1.color * cosine_factor * sin(80 * fy); /* with sinusoidal stripey pattern */
	else if (rendermode == 5) output[work_item_id] = sphere1.color * cosine_factor * sin(400 * fy) * sin(400 * fx); /* with grid pattern */
	else output[work_item_id] = normal * 0.5f + (float3)(0.5f, 0.5f, 0.5f); /* with normal colours */
}