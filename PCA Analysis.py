import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成示例數據
np.random.seed(42)
data = {
    'feature1': np.random.rand(100) * 100,
    'feature2': np.random.rand(100) * 100,
    'feature3': np.random.rand(100) * 100,
    'feature4': np.random.rand(100) * 100
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 數據標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 應用 PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])

# 繪製 PCA 結果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='principal_component_1', y='principal_component_2', data=df_pca, palette='viridis')
plt.title('PCA Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca_analysis.png')
plt.show()

# 輸出解釋方差比率
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)