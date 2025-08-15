# ElevateLabTask7

# SVM Classification (Simple Version)

## Steps
1. Load Breast Cancer dataset
2. Scale features
3. Train SVM with:
   - Linear kernel
   - RBF kernel
4. Show accuracy & cross-validation score
5. Plot decision boundaries (first 2 features)

| # | Question                                      | Simple Answer                                                                                             |
| - | --------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 1 | **What is a support vector?**                 | Points that are closest to the decision line. They help decide the position of that line.                 |
| 2 | **What does the C parameter do?**             | Controls mistakes in training. Large C = less mistakes, smaller C = wider margin but maybe more mistakes. |
| 3 | **What are kernels in SVM?**                  | Functions that change data shape so it’s easier to separate.                                              |
| 4 | **Difference between Linear and RBF kernel?** | Linear → straight line boundary; RBF → curved boundary for complex data.                                  |
| 5 | **Advantages of SVM?**                        | Works well for small-to-medium data, good in high dimensions, and often accurate.                         |
| 6 | **Can SVMs be used for regression?**          | Yes, called SVR .                                                              |
| 7 | **What if data is not linearly separable?**   | Use kernels  to separate in higher dimensions.                                                  |
| 8 | **How is overfitting handled in SVM?**        | Tune C and gamma, use cross-validation to check performance.                                              |

