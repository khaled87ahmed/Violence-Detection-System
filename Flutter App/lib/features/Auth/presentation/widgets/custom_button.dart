import 'package:flutter/material.dart';
import 'package:security_app/constant.dart';

class CustomButton extends StatelessWidget {
  const CustomButton({required this.onPressed, required this.buttonName});
  final VoidCallback onPressed;
  final String buttonName;
  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor:Colors.white ,
          foregroundColor: kPrimaryColor,
        ),
        
        onPressed: onPressed,
        child: Text(buttonName),
      ),
    );
  }
}
