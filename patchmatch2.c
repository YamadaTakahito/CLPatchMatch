unsigned int getIndex(const int y, const int x, const int z, const int width);
double dis(
	__global double *img1, const int y1, const int x1,
	__global double *img2, const int y2, const int x2,
	const int hpatchWidth,
	const int width);
int random(int start, int end, unsigned int seed);

unsigned int getIndex(const int y, const int x, const int z, const int width)
{
	return ((y * width + x) * 3 + z);
}

double dis(
	__global double *img1, const int y1, const int x1,
	__global double *img2, const int y2, const int x2,
	const int hpatchSize,
	const int width)
{
	int yy1 = y1 + hpatchSize;
	int yy2 = y2 + hpatchSize;
	int xx1 = x1 + hpatchSize;
	int xx2 = x2 + hpatchSize;

	double diff = 0;
	for (int j = -hpatchSize; j < hpatchSize; ++j)
		for (int i = -hpatchSize; i < hpatchSize; ++i)
			for (int k = 0; k < 3; ++k)
			{
				double t = (img1[getIndex(j + yy1, i + xx1, k, width)] - img2[getIndex(j + yy2, i + xx2, k, width)]);
				diff += t * t;
			}
	return diff;
}

int random(int start, int end, unsigned int seed)
{
	unsigned int num = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	return num % (end - start + 1) + start;
}

#define D(y1, x1, y2, x2) dis(img1, (y1), (x1), img2, y2, x2, hpatchSize, width + hpatchSize)

#define nff(i, j, k) output[((i)*width + (j)) * 3 + (k)]

#define MAXINT 9999999.0

__kernel void randomfill(const int height, const int width,
						 const int min_offset, const int max_offset,
						 const int hpatchSize,
						 __global double *img1, __global double *img2, __global double *output)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int seed = y << 16 + x;

	int minX, minY, maxX, maxY;
	// 中心より大きい
	if (height / 2 < y)
	{
		minY = max(0, y - max_offset);
		maxY = max(0, y - min_offset);
	}
	else
	{
		maxY = min(height - 1, y + max_offset);
		minY = min(height - 1, y + min_offset);
	}

	if (width / 2 < x)
	{
		minX = max(0, x - max_offset);
		maxX = max(0, x - min_offset);
	}
	else
	{
		maxX = min(width - 1, x + max_offset);
		minX = min(width - 1, x + min_offset);
	}

	int ty = seed = nff(y, x, 0) = random(minY, maxY, seed);
	int tx = nff(y, x, 1) = random(minX, maxX, seed);
	nff(y, x, 2) = D(y, x, ty, tx);
}

__kernel void propagate(const int hpatchSize, const int height, const int width,
						const min_offset, const max_offset, const int iteration,
						__global double *img1, __global double *img2, __global double *output)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int direction = 1;
	if (iteration % 2 == 0)
	{
		x = width - x - 1;
		y = height - y - 1;
		direction = -1;
	}

	// PROPAGATION START
	int dir = direction; //temp direction
	if (direction == -1)
		if (y + 1 >= height || x + 1 >= width)
			return;

	double currentD, topD, leftD;
	//compute intensitive part

	dir = direction;
	if (nff(y - dir, x, 0) + dir >= height || nff(y - dir, x, 0) + dir < 0)
	{
		topD = MAXINT;
	}
	else
	{
		int y2 = nff(y - dir, x, 0);
		int x2 = nff(y - dir, x, 1);
		int offsetx = abs(x - x2);
		int offsety = abs(y - y2);
		if (offsetx >= min_offset && offsetx <= max_offset && offsety >= min_offset && offsety <= max_offset)
		{
			topD = D(y, x, y2 + dir, x2);
		}
		else
		{
			topD = MAXINT;
		}
	}

	if (nff(y, x - dir, 1) + dir >= width || nff(y, x - dir, 1) + dir < 0)
	{
		leftD = MAXINT;
	}
	else
	{
		int y2 = nff(y, x - dir, 0);
		int x2 = nff(y, x - dir, 1);
		int offsetx = abs(x - x2);
		int offsety = abs(y - y2);
		if (offsetx >= min_offset && offsetx <= max_offset && offsety >= min_offset && offsety <= max_offset)
		{
			leftD = D(y, x, y2, x2 + dir);
		}
		else
		{
			leftD = MAXINT;
		}
	}

	dir = direction;
	currentD = nff(y, x, 2);

	if (topD < currentD)
	{
		nff(y, x, 0) = nff(y - dir, x, 0) + dir;
		nff(y, x, 1) = nff(y - dir, x, 1);
		currentD = nff(y, x, 2) = topD;
	}

	if (leftD < currentD)
	{
		nff(y, x, 0) = nff(y, x - dir, 0);
		nff(y, x, 1) = nff(y, x - dir, 1) + dir;
		currentD = nff(y, x, 2) = leftD;
	}
	// PROPAGATION END

	//random search
	unsigned int seed = 1;
	int w = width, h = height;

	while (h > 1 && w > 1)
	{
		int x1, y1, x2, y2;
		y1 = y - h / 2;
		x1 = x - w / 2;
		y2 = y + h / 2;
		x2 = x + w / 2;
		x1 = max(0, x1);
		y1 = max(0, y1);
		x2 = min(width - 1, x2);
		y2 = min(height - 1, y2);

		int targetX = seed = random(x1, x2, seed);
		int targetY = seed = random(y1, y2, seed);
		int offsetx = abs(x - targetX);
		int offsety = abs(y - targetY);

		if (offsetx >= min_offset && offsetx <= max_offset && offsety >= min_offset && offsety <= max_offset)
		{
			double newD = D(y, x, targetY, targetX);
			if (newD < nff(y, x, 2))
			{
				nff(y, x, 0) = targetY;
				nff(y, x, 1) = targetX;
				nff(y, x, 2) = newD;
			}
		}
		w /= 2;
		h /= 2;
	}
}