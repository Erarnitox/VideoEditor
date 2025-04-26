<a id="top"></a>
<br />
<div align="center">
  <h3 align="center">Erarnitox's Video Editor</h3>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About The Project</a>
    </li>
    <li>
      <a href="#build">Getting Started</a>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

This is a very simple video editor that automatically cuts and speeds up sections of a lecture video to make it shorter, cleaner and easier to digest.
I have especially created this tool to aid me in cutting my own youtube videos.

Here's why:
* Cutting long tutorial videos takes ages and is boring
* You can save a lot of time by automating the busy work

Of cause this editor isn't perfect, but it can help to do the heavy lifting.


<p align="right">(<a href="#top">back to top</a>)</p>


## Build
First you need to clone this repository

```
git clone https://github.com/Erarnitox/VideoEditor
```

#### Prerequisites
You need to have cmake, clang, ninja and ffmpeg installed!

#### Actually Building

Once downloaded you can follow these instructions to build the video editor

```
cd videoEditor && cmake --preset default && cmake --build build
```

this will produce a `video_editor` binary in the `build` directory.

## Usage

Simple usage:
The first command line argument is the input video and the second command line argument is the output video.

```
./build/video_editor input.mp4 output.mp4
```

### Options
You can supply additional command line options:
```
--silent-speed <speedValue>
--silent-threshold <threshold>
--min-silence <minSilenceDuration>
--visual-diff <visualDiffThreshold>
--min-inhale <minInhaleDurationMs>
--max-inhale <maxInhaleDurationMs>
--inhale-low <inhaleLowThreshold>
--inhale-high <inhaleHighThreshold>
--debug <debugMode>
--watermark <watermarkString>
```

<p align="right">(<a href="top">back to top</a>)</p>

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


## License

Distributed under the MIT License.

<p align="right">(<a href="#top">back to top</a>)</p>


## Contact

Erarnitox - (I'm known as @erarnitox on X and Discord! Find me there!)

Project Link: [https://github.com/Erarnitox/VideoEditor](https://github.com/Erarnitox/VideoEditor)

<p align="right">(<a href="#top">back to top</a>)</p>