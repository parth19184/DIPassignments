import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def main():
    def show_grey_image(image_matrix):
        plt.imshow(image_matrix, cmap='gray')
        plt.show()

    def get_probability_list(matrix):
        input_probability_list = []
        sum_here = 0
        for i in range(256):
            sum_here += (matrix == i).sum()
            input_probability_list.append(((matrix == i).sum())/65536)
        return input_probability_list
    def get_cdf_list(input_probability_list):
        input_cdf_list = []
        sum_iter = 0
        for i in range(256):
            sum_iter += input_probability_list[i]
            input_cdf_list.append(sum_iter)
        return input_cdf_list

    def plot_histogram(x_axis_list, y_axis_list, x_axis_name, y_axis_name, plot_name):
        plt.bar(x_axis_list, y_axis_list)
        plt.title(plot_name)
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.show()
    
    def swap_columns():
        pass
    
    def swap_rows():
        pass
    
    def rotate_matrix(matrix):
        (m,n) = matrix.shape
        col_inverse_matrix =np.zeros((m,n))
        row_inverse_matrix = np.zeros((m,n))
        for i in range(n):
            col_inverse_matrix[:,i] = matrix[:, n - 1 - i]
        for i in range(m):
            row_inverse_matrix[i] = col_inverse_matrix[n - 1 - i]
        return row_inverse_matrix.astype(int)

    def mul_result(matrix1, matrix2):
        (m,n)  =matrix1.shape
        answer = 0
        for i in range(m):
            for j in range(n):
                answer += matrix1[i][j]*matrix2[i][j]

        return answer

    def question3():
        input_image_path = input("write input image path:")
        image_here = cv2.imread(input_image_path)
        matrix = plt.imread(input_image_path, 0)
        matrix = np.array(matrix)
        show_grey_image(matrix)
        
        input_probability_list = get_probability_list(matrix)
        input_cdf_list = get_cdf_list(input_probability_list)
        intensity_list = np.arange(0,256)

        plot_histogram(intensity_list, input_probability_list, 'intensity values', 'probability', 'probabilities of intensities')
        print('probability histogram plotted')
        plot_histogram(intensity_list, input_cdf_list, 'intensity values', 'CDF', 'CDF of intensities')
        print('cdf plotted')
        output_matrix = np.ones((256,256))* -1
        for i in intensity_list:
            output_matrix = np.where(matrix == i, 255*input_cdf_list[i], output_matrix)     #for replacing intensity values with 255*F(r)

        output_probability_list = get_probability_list(output_matrix)
        show_grey_image(output_matrix)
        plot_histogram(intensity_list, output_probability_list, 'intensity values', 'probability', 'probabilities of intensities in output image')
    
    def question4():
        input_image_path = input("write input image path:")
        image_here = cv2.imread(input_image_path)
        matrix = plt.imread(input_image_path, 0)
        matrix = np.array(matrix)
        show_grey_image(matrix)
        print("original image shown")
        gamma_value = float(input("enter gamma value for image:"))
        intensity_list = np.arange(0,256)
        #target_matrix = np.ones((256,256))* -1
        '''for i in intensity_list:
            target_matrix = np.where(matrix == i, 255*((i/255)**gamma_value), target_matrix)'''

        target_matrix = np.array(255*(matrix/255)**gamma_value).astype(int)
        #print(target_matrix)
        show_grey_image(target_matrix)
        print("gamma corrected image shown")

        input_probability_list = get_probability_list(matrix)
        input_cdf_list = get_cdf_list(input_probability_list)

        target_probability_list = get_probability_list(target_matrix)
        target_cdf_list = get_cdf_list(target_probability_list)
        #print(target_probability_list)
        plot_histogram(intensity_list, input_probability_list, 'intensity values', 'probabilities', 'Histogram of input image')
        print("input image histogram plotted")
        plot_histogram(intensity_list, target_probability_list, 'intensity values', 'probabilities', 'Histogram of target image')
        print("target image histogram plotted")

        plot_histogram(intensity_list, input_cdf_list, 'intensity values', 'cdf F(r)', 'CDF of input image')
        print("input image CDF plotted")
        plot_histogram(intensity_list, target_cdf_list, 'intensity values', 'cdf G(s)', 'CDF of target image')
        print("target image CDF plotted")

        mapping_list = []
        target_cdf_list = np.array(target_cdf_list)
        for i in intensity_list:
            mapping_list.append(np.argmin(np.abs(input_cdf_list[i] - target_cdf_list)))

        print(mapping_list)

        output_matrix = np.ones((256,256)) * -1
        for i in intensity_list:
            output_matrix = np.where(matrix == i, mapping_list[i], output_matrix)
        show_grey_image(output_matrix)
        output_probability_list = get_probability_list(output_matrix)
        plot_histogram(intensity_list,output_probability_list, 'intensity values', 'probabilities', 'matched image histogram')
        
    def question5():
        random_matrix= np.random.randint(1,10,size=(3,3))
        random_filter = np.random.randint(1,10,size=(3,3))
        print(random_filter)
        rotated_filter = rotate_matrix(random_filter)

        padded_matrix = np.zeros((7,7))
        padded_matrix[2:5][:,2:5] = random_matrix
        print(padded_matrix)
        answer_matrix = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                first_matrix = padded_matrix[i:i + 3][:,j:j + 3]
                answer_matrix[i][j] = mul_result(first_matrix, rotated_filter)

        print(answer_matrix)
    input_question = input("enter question number:")
    exec('question{}()'.format(input_question))

if __name__ == '__main__':
    main()