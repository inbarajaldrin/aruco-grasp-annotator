# ArUco Marker PDF Generator

This directory contains scripts for generating printable PDFs with ArUco markers using the same specifications as the main ArUco annotator application.

## Scripts

### `generate_aruco_pdf.py`

Main script for generating PDFs with ArUco markers. Supports customizable marker sizes, border widths, and layouts.

#### Usage

```bash
# Basic usage - generate markers 0-5 with default settings
# PDF will be saved in the same folder as the script
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py

# Custom marker IDs and size
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py \
    --output my_markers.pdf \
    --ids 0 1 2 3 4 5 \
    --size 30 \
    --border 20

# Large markers for better visibility
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py \
    --output large_markers.pdf \
    --ids 0 1 2 3 4 5 \
    --size 50 \
    --border 20 \
    --per-row 2

# Small markers for compact applications
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py \
    --output small_markers.pdf \
    --ids 0 1 2 3 4 5 \
    --size 20 \
    --border 20 \
    --per-row 4
```

#### Command Line Options

- `--output, -o`: Output PDF file path (default: `aruco_markers.pdf` in the scripts folder)
- `--size, -s`: Marker size in millimeters (default: 30)
- `--border, -b`: Border width percentage (default: 20)
- `--ids, -i`: Marker IDs to generate (default: 0 1 2 3 4 5)
- `--dictionary, -d`: ArUco dictionary (default: DICT_4X4_50)
- `--page-size`: Page size - A4 or letter (default: A4)
- `--margin`: Page margin in mm (default: 20)
- `--per-row`: Markers per row (default: 3)
- `--black-border`: Black border width in mm (default: 2)
- `--no-black-border`: Disable black border around markers


## Features

### Marker Specifications

- **Same as App**: Uses identical specifications to the main ArUco annotator application
- **Configurable Size**: Marker size from 10mm to 100mm
- **Border Control**: White border width as percentage of marker size (0-50%)
- **Black Border**: Optional black border around each marker for better contrast
- **High Quality**: 4 pixels per millimeter resolution for crisp printing

### Layout Options

- **Flexible Layout**: Automatic layout calculation based on page size and margins
- **Customizable Spacing**: Control markers per row and spacing
- **Multiple Page Sizes**: Support for A4 and Letter paper sizes
- **Professional Layout**: Includes title, specifications, and marker ID labels

### ArUco Support

- **Multiple Dictionaries**: Support for all ArUco dictionaries
- **Custom IDs**: Generate any marker IDs within dictionary limits
- **DICT_4X4_50 Default**: Uses the same dictionary as the main app

## Examples

### Standard Configuration (Same as App)

```bash
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py \
    --output standard_markers.pdf \
    --ids 0 1 2 3 4 5 \
    --size 30 \
    --border 20 \
    --dictionary DICT_4X4_50
```

### Large Markers for Robotics

```bash
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py \
    --output robotics_markers.pdf \
    --ids 0 1 2 3 4 5 6 7 8 9 \
    --size 50 \
    --border 20 \
    --per-row 2 \
    --black-border 3
```

### Small Markers for Compact Applications

```bash
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py \
    --output compact_markers.pdf \
    --ids 0 1 2 3 4 5 6 7 8 9 10 11 \
    --size 20 \
    --border 20 \
    --per-row 4 \
    --black-border 1
```

## Dependencies

- `reportlab>=4.0.0` - PDF generation
- `opencv-python>=4.8.0` - ArUco marker generation
- `numpy>=1.24.0` - Numerical operations

## Integration with Main App

The PDF generator uses the same `ArUcoGenerator` class as the main application, ensuring:

- **Consistent Markers**: Identical ArUco patterns
- **Same Specifications**: Matching dimensions and border widths
- **Compatible Output**: Markers work seamlessly with the 3D annotator

## Printing Tips

1. **Use High Quality Paper**: Print on white, matte paper for best results
2. **Check Print Settings**: Ensure "Actual Size" or "100%" scaling
3. **Verify Dimensions**: Measure printed markers to confirm correct size
4. **Black Border**: The optional black border improves detection reliability
5. **Cutting**: Use a sharp blade to cut markers precisely along the black border

## Troubleshooting

### Common Issues

1. **Module Not Found**: Ensure you're using `uv run` to activate the environment
2. **PDF Not Generated**: Check file permissions in the output directory
3. **Poor Print Quality**: Increase marker size or use higher DPI settings
4. **Layout Issues**: Adjust `--per-row` and `--margin` parameters

### Getting Help

```bash
# Show help
uv run python src/aruco_annotator/scripts/generate_aruco_pdf.py --help
```
