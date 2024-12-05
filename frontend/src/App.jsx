import { useState } from "react";
import { Box, Button, Image, Input, Text } from "@chakra-ui/react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please upload a valid image file.");
        return;
      }
      setError(null);
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAddInClick = async () => {
    if (!image) {
      setError("Please upload an image before processing.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "http://localhost:8000/transfer",
        { image },
        {
          headers: { "Content-Type": "application/json" },
          responseType: "arraybuffer",
        }
      );

      const blobResponse = new Blob([response.data], { type: "image/png" });
      const imageUrl = URL.createObjectURL(blobResponse);
      setOutputImage(imageUrl);
    } catch (err) {
      if (err.response) {
        const message = new TextDecoder("utf-8").decode(err.response.data);
        setError(`Error from server: ${message}`);
      } else {
        setError("An error occurred while processing the image.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="center"
      flexDirection="column"
      className="mainpage"
      gap="40px"
    >
      <Text
        as="h1"
        fontSize="5xl"
        fontWeight="bold"
        css={{
          background:
            "linear-gradient(to right, red, orange, yellow, green, blue, purple)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        Diffusion Colorization
      </Text>
      {error && (
        <Text color="red.500" fontSize="sm">
          {error}
        </Text>
      )}
      <Box display="flex" gap="20px" flexWrap="wrap" justifyContent="center">
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          border="2px solid lightgray"
          borderRadius="10px"
          padding="20px"
          cursor="pointer"
          className="output"
          onClick={() => document.getElementById("fileInput").click()}
        >
          <Input
            id="fileInput"
            type="file"
            accept="image/*"
            display="none"
            onChange={handleImageUpload}
          />
          <Text marginBottom="10px">Upload an Image</Text>
          {image && (
            <Image
              src={image}
              alt="Uploaded Preview"
              boxSize="200px"
              objectFit="cover"
            />
          )}
          <Text fontSize="sm" color="gray.500">
            {image ? "Uploaded" : "Click to upload"}
          </Text>
        </Box>
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          border="2px solid lightgray"
          borderRadius="10px"
          padding="20px"
          className="output"
        >
          <Text marginBottom="10px">Output Image</Text>
          {outputImage ? (
            <Image
              src={outputImage}
              alt="Output"
              boxSize="200px"
              objectFit="cover"
            />
          ) : (
            <Text fontSize="sm" color="gray.500">
              {loading ? "Processing..." : "No output yet"}
            </Text>
          )}
        </Box>
      </Box>
      <Button
        size={"2xl"}
        style={{
          backgroundColor: "#2b6cb0",
          color: "white",
          borderRadius: "10px",
        }}
        onClick={handleAddInClick}
        isLoading={loading}
        loadingText="Processing"
      >
        Call Model
      </Button>
    </Box>
  );
}

export default App;
