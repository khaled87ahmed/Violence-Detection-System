import 'package:flutter/material.dart';
import 'package:security_app/constant.dart';

void snackbarMessage(BuildContext context, String message) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      backgroundColor: Colors.white,
      content: Text(
        message,
        style: TextStyle(color: kPrimaryColor),
      ),
    ),
  );
}
