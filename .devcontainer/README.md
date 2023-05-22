## VS Code Dev Container

> 无 GUI 运行 Instant Neural Graphics Primitives 的基本开发容器。

### 要求

-   #### **[Docker](https://www.docker.com/get-started)**

-   #### **[VS Code](https://code.visualstudio.com/Download)**

-   #### **[Docker VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)**

### 构建步骤

```sh
cmake -DNGP_BUILD_WITH_GUI=off ./ -B ./build
cmake --build build --config RelWithDebInfo -j 16
```
