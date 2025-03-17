'''
script was comparing the vectorized kernel estimation
'''
# from stanley_appex.utils import *
def kernel_vec(x, y, t_curr, t_future, A, H):
    # matrix kernel between (x, t_curr) and (y, t_future)
    m = (expm(A*(t_future - t_curr))@(x.T)).T # e^(Adt) x
    d = y[:, None, :] - m[None, :, :]  # Shape: (M, N, 2)
    c = cov(A, H, t_curr, t_future) # cov

    result = np.exp(-np.einsum('mni,ij,mnj->nm', d, np.linalg.inv(c), d))
    return result # np.exp(-0.5*np.einsum('ijk, ijl, kl -> ij', d, d, np.linalg.inv(c))) # returns N x N kernel

def for_loop(x, y, t_curr, t_future, A, H):
    c = cov(A, H, t_curr, t_future)
    result = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            m = expm(A*(t_future - t_curr))@x[i]
            d = y[j] - m
            result[i, j] = np.exp(-
                d.T @ np.linalg.inv(c) @ d
            )
    return result

idx=111
ticurr, tifuture = idx, idx+1
t_curr, t_future = ts_data[ticurr], ts_data[tifuture]
x = np.array(xs_data[ticurr])
y = np.array(xs_data[tifuture])
K_rect = kernel_vec(x, y, t_curr, t_future, A, H)
K_square, Kest = pairwise_kernel(xs_data_downsampled, ts_data_downsampled, ticurr, A, H)
K_loop = for_loop(x, y, t_curr, t_future, A, H)

# print(np.linalg.norm(K_square - K_rect))

plt.imshow(K_rect), plt.title("rect"), plt.colorbar(), plt.show()
plt.imshow(K_square), plt.title("square"), plt.colorbar(), plt.show()
plt.imshow(K_loop), plt.title("loop"), plt.colorbar(), plt.show()

K_square, K_rect