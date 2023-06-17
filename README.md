# Smart Server-Side API

This API is built using Flask framework and is responsible for handling the server-side processing of images for computer vision and image processing tasks. It provides two main routes: `/upload` and `/calculate`.

## Installation

1. Clone the repository:

```shell
git clone <repository_url>
```
2. Install the required dependencies:

```shell
pip install -r requirements.txt
```
3. Run the server:

```shell
python app.py
```
4. Run the server:
The server will now be running on ```http://localhost:5000.```

5. Expose the server to the internet using ngrok:
   - Install ngrok by following the instructions at [ngrok.com](https://ngrok.com).
   - Run the following command to expose the local server:
     ```
     ngrok http 5000
     ```
   - Note the forwarding URL provided by ngrok (e.g., `http://abcd1234.ngrok.io`). This URL will be used for client-side API calls.


## Notes
- Ensure that the uploaded image is in a supported format (e.g., JPEG, PNG) and contains a clear representation of the equation.
- The image processing and equation recognition algorithms are implemented using OpenCV and other computer vision techniques. Adjustments or improvements   to these algorithms can be made in the corresponding code files.

## Routes

` /upload - POST `
This route is used to upload an image to the server for processing. It expects a file named `file` in the request form data. Upon successful upload, it will return the equation as a JSON response.

` /calculate - GET`
This route is used to calculate the solution for a given equation. It expects the equation as a query parameter named `equation`. It will return the solution as a JSON response.

## architect
![Arch](https://user-images.githubusercontent.com/100499106/246194668-1c70a853-5f89-4d52-8c20-5269024b4e60.png)

## Client-Side Application

For the client-side application code and detailed explanations, please refer to the [Smooty](https://github.com/Dris7/Smoorty). The repository contains the source code for the mobile app along with comprehensive instructions on installation, usage, and additional features.

In the client-side repository, you will find:

- Codebase for the mobile app developed using [flutter/java].
- Step-by-step instructions for setting up the development environment.
- Detailed documentation on app functionality, user interfaces, and implementation details.
- Guides on how to integrate the client-side with the server-side API.
  
To access the client-side repository, click on the following link: 
[Smooty_Java_Android](https://github.com/Dris7/Smoorty)
[Smooty_Flutter_iOs_Android](https://github.com/Dris7/Smoorty_flutter)

Feel free to explore the repository, contribute to the code, and raise any issues or questions you may have.
## License
This project is licensed under the MIT License.

