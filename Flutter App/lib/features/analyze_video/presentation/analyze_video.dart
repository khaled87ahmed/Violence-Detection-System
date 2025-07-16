import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:socket_io_client/socket_io_client.dart' as IO;
import 'package:url_launcher/url_launcher.dart';

class AnalyzeVideoView extends StatefulWidget {
  const AnalyzeVideoView({super.key});

  @override
  State<AnalyzeVideoView> createState() => _AnalyzeVideoViewState();
}

class _AnalyzeVideoViewState extends State<AnalyzeVideoView> {
  File? selectedFile;
  bool isProcessing = false;
  double progress = 0.0;
  String? processedVideoUrl;
  double? violencePercentage;
  String serverUrl = "http://192.168.100.3:5000"; // <-- عدل حسب سيرفرك

  late IO.Socket socket;

  @override
  void initState() {
    super.initState();
    socket = IO.io(serverUrl, <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': false,
    });

    socket.connect();

    socket.on('processing_progress', (data) {
      setState(() {
        progress = data['progress'];
      });
    });

    socket.on('processing_complete', (data) {
      setState(() {
        isProcessing = false;
        processedVideoUrl = "$serverUrl${data['processed_video']}";
        violencePercentage = data['violence_percentage'];
      });
    });

    socket.on('processing_error', (_) {
      setState(() {
        isProcessing = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("فشل في تحليل الفيديو")),
      );
    });
  }

  Future<void> pickVideo() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.video);
    if (result != null && result.files.single.path != null) {
      setState(() {
        selectedFile = File(result.files.single.path!);
        processedVideoUrl = null;
        violencePercentage = null;
      });
    }
  }

  Future<void> uploadVideo() async {
    if (selectedFile == null) return;
    setState(() {
      isProcessing = true;
      progress = 0;
    });

    var request = http.MultipartRequest("POST", Uri.parse('$serverUrl/upload'));
    request.files.add(
      await http.MultipartFile.fromPath("file", selectedFile!.path),
    );

    var response = await request.send();

    if (response.statusCode != 200) {
      setState(() => isProcessing = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("فشل في رفع الفيديو")),
      );
    }
  }

  void openProcessedVideo() async {
    if (processedVideoUrl != null) {
      final url = Uri.parse(processedVideoUrl!);
      if (await canLaunchUrl(url)) {
        await launchUrl(url, mode: LaunchMode.externalApplication);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("لا يمكن فتح الفيديو")),
        );
      }
    }
  }

  @override
  void dispose() {
    socket.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Analyze Video"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: pickVideo,
              child: Center(child: Text("اختيار فيديو")),
            ),
            if (selectedFile != null) ...[
              SizedBox(height: 10),
              Text("تم اختيار: ${selectedFile!.path.split('/').last}"),
              SizedBox(height: 10),
              ElevatedButton(
                onPressed: isProcessing ? null : uploadVideo,
                child: Text("بدء التحليل"),
              ),
            ],
            if (isProcessing) ...[
              SizedBox(height: 20),
              LinearProgressIndicator(value: progress / 100),
              SizedBox(height: 10),
              Text("جاري المعالجة... ${progress.toStringAsFixed(1)}%"),
            ],
            if (violencePercentage != null) ...[
              SizedBox(height: 20),
              Text(
                "نسبة العنف في الفيديو: ${violencePercentage!.toStringAsFixed(2)}%",
                style: TextStyle(fontSize: 18, color: Colors.red),
              ),
              SizedBox(height: 10),
              ElevatedButton(
                onPressed: openProcessedVideo,
                child: Text("عرض الفيديو المعالج"),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
