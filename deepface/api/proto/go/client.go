package main

import (
	"context"
	"flag"
	"log"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/dal-papa/deepface/proto"
)

var flagImg = flag.String("img", "", "Path to the image file")
var flagImg2 = flag.String("img2", "", "Path to the second image file for verification")
var flagMode = flag.String("mode", "analyze", "Mode of operation: analyze, detect, or verify")

func main() {
	ctx := context.Background()
	flag.Parse()
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewDeepFaceServiceClient(conn)
	if *flagMode == "analyze" {

		res, err := client.Analyze(ctx, &pb.AnalyzeRequest{
			ImageUrl: *flagImg,
			Actions: []pb.AnalyzeRequest_Action{
				pb.AnalyzeRequest_AGE,
				pb.AnalyzeRequest_GENDER,
				pb.AnalyzeRequest_RACE,
			},
		})
		if err != nil {
			log.Fatalf("error calling Analyze: %v", err)
		}
		log.Printf("Analyze response: %+v\n", res)
	} else if *flagMode == "verify" {
		res, err := client.Verify(ctx, &pb.VerifyRequest{
			Image1Url: *flagImg,
			Image2Url: *flagImg2,
		})
		if err != nil {
			log.Fatalf("error calling Verify: %v", err)
		}
		log.Printf("Photo is verified: %t\n", res.Verified)
		log.Printf("Verify response: %+v\n", res)
	} else if *flagMode == "represent" {
		res, err := client.Represent(ctx, &pb.RepresentRequest{
			ImageUrl: *flagImg,
		})
		if err != nil {
			log.Fatalf("error calling Represent: %v", err)
		}
		for _, rep := range res.Results {
			log.Printf("Representation: %+v\n", rep)
		}
	} else {
		log.Fatalf("Unknown mode: %s", *flagMode)
	}
}
