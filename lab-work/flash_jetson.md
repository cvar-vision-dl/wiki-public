
# Jetson flash guide (Jetpack 5.1.2)

## Requirements
- Ubuntu 20.04 (or use docker [NOT TESTED])

## Step 1: Install SDK Manager
Download SDK Manager from [the official website](https://developer.nvidia.com/sdk-manager) and install it following the steps in the same website.

## Step 2: Run SDK Manager and download Jetpack 5.1.2
Run SDK Manager

### Step 2.1: Run SDK Manager
Run with the command (recommended):
```
sdkmanager --cli
```
or
```
sdkmanager
```
> [!NOTE]
> We recommend the use of --cli option as everything is done directly in the terminal

### Step 2.2: Select option 'Download only' (or 'Download now. Install later' in not cli mode)

![download-only](/assets/jetson-download-only.png)

or

![download-only-not-cli](https://docs.nvidia.com/sdk-manager/_images/sdkm-2-download-install-options-jetson.03.png)

### Step 2.3: Download Jetpack 5.1.2
Select the product (Jetson), the target hardware model, Jetpack 5.1.2, DeepStream (if needed), download route...
Once all is correct, proceed with the download.

### Step 2.4 (! ONLY for SSD usage in Jetson): Fix config file
Once Jetpack 5.1.2 is installed correctly, navigate to the folder (~/nvidia/nvidia_sdk/JetPack_5.1.2_Linux_.../Linux_for_Tegra/tools/kernel_flash/)

![jetson-folder](/assets/jetson-folder.png)

Copy the contents of file 'flash_l4t_external.xml' to both 'flash_l4t_nvme.xml' and 'flash_l4t_t234_nvme.xml':

```
sudo cp flash_l4t_external.xml flash_l4t_nvme.xml && sudo cp flash_l4t_external.xml flash_l4t_t234_nvme.xml
```

## Step 3: Install Jetpack 5.1.2
Now select 'Install' option and follow the instructions of the app.

> [!NOTE]
> If you use manual mode, remember to remove the jumper cable from pins when the SDK Manager already detected the Jetson, just before installing.
